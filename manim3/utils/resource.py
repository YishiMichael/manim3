from typing import Callable, Generic, TypeVar


__all__ = [
    "lazy_property",
    "ResourceObject"
]


# TODO: typing disaster
T = TypeVar("T")


class LazyPropertyData(Generic[T]):
    def __init__(self, value: T):
        self.value: T = value
        self.requires_update: bool = False
        #affected_properties: list[str]
        #requires_update: bool = True
        #value: T | None = None


DEPENDENCY_DICT: dict[str, dict[str, set[str]]] = {}
CURRENTLY_INITIALIZING: dict[str, list[str]] = {}


class ResourceObject:
    def __init__(self):
        self._lazy_properties_: dict[str, LazyPropertyData] = {}
        #self.current_initializing: list[str] = []
        cls_name = self.__class__.__name__
        DEPENDENCY_DICT[cls_name] = {}
        CURRENTLY_INITIALIZING[cls_name] = []

    def __getattribute__(self, name):
        result = super().__getattribute__(name)
        if isinstance(result, lazy_property):
            cls_name = self.__class__.__name__
            if CURRENTLY_INITIALIZING[cls_name]:
                DEPENDENCY_DICT[cls_name][name].update(CURRENTLY_INITIALIZING[cls_name])
        return result

    def __init_subclass__(cls) -> None:
        pass  # TODO: init CURRENTLY_INITIALIZING, etc here

    def set_lazy_property(self, name, value):
        self._lazy_properties_[name].value = value
        cls_name = self.__class__.__name__
        for affected_name in DEPENDENCY_DICT[cls_name][name]:
            self._lazy_properties_[affected_name].requires_update = True
        return self


class lazy_property(property):
    def __new__(cls, func: Callable):
        name = func.__name__

        def f_get(obj):
            #assert isinstance(obj, ResourceObject)
            if name not in obj._lazy_properties_:
                cls_name = obj.__class__.__name__
                DEPENDENCY_DICT[cls_name][name] = set()
                CURRENTLY_INITIALIZING[cls_name].append(name)
                value = func(obj)
                CURRENTLY_INITIALIZING[cls_name].remove(name)
                obj._lazy_properties_[name] = LazyPropertyData(value)
            else:
                property_data = obj._lazy_properties_[name]
                if property_data.requires_update:
                    value = func(obj)
                    property_data.value = value
                    property_data.requires_update = False
                else:
                    value = property_data.value
            return value

        return property(f_get)


class A(ResourceObject):
    @lazy_property
    def a(self):
        return 0

    @a.setter
    def a(self, value):
        self.set_lazy_property("a", value)

    @lazy_property
    def b(self):
        return self.a + 1


a = A()
print(a.a)
a.a = 1
print(a.b)
print()
