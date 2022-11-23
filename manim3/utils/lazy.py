#from typing import Callable, Generic, TypeVar


__all__ = [
    "lazy_property",
    "LazyMeta"
]


# TODO: typing disaster


class LazyPropertyData:
    def __init__(self, value):
        self.value = value
        self.requires_update: bool = False


class lazy_property(property):
    pass


class LazyMeta(type):
    def __new__(cls, cls_name, bases, attrs):
        cls._currently_initializing_: list[str] = []
        cls._dependency_dict_: dict[str, set[str]] = {}

        new_attrs = {}
        for name, attr in attrs.items():
            if not isinstance(attr, lazy_property):
                new_attrs[name] = attr
                continue
            if attr.fget is None:
                continue

            cls._dependency_dict_[name] = set()

            getter = cls.setup_lazy_getter(name, attr.fget)
            if attr.fset is not None:
                setter = cls.setup_lazy_setter(name)
            else:
                setter = cls.setup_deleted_setter()
            deleter = cls.setup_deleted_deleter()
            new_attrs[name] = lazy_property(getter, setter, deleter)

        __init__ = attrs.get("__init__", None)
        def new_init(self, *args, **kwargs):
            self._lazy_properties_ = {}
            if __init__ is not None:
                __init__(self, *args, **kwargs)
        new_attrs["__init__"] = new_init

        return super().__new__(cls, cls_name, bases, new_attrs)

    @classmethod
    def setup_lazy_getter(cls, name, fget):
        def new_fget(self):
            if self.__class__._currently_initializing_:
                self.__class__._dependency_dict_[name].update(self.__class__._currently_initializing_)
            if name not in self._lazy_properties_:
                self.__class__._currently_initializing_.append(name)
                value = fget(self)
                self.__class__._currently_initializing_.remove(name)
                self._lazy_properties_[name] = LazyPropertyData(value)
            else:
                property_data = self._lazy_properties_[name]
                if property_data.requires_update:
                    value = fget(self)
                    property_data.value = value
                    property_data.requires_update = False
                else:
                    value = property_data.value
            return value
        return new_fget

    @classmethod
    def setup_lazy_setter(cls, name):
        def new_fset(self, value):
            self._lazy_properties_[name].value = value
            for affected_name in self.__class__._dependency_dict_[name]:
                self._lazy_properties_[affected_name].requires_update = True
            #fset(self, value)
        return new_fset

    @classmethod
    def setup_deleted_setter(cls):
        def new_fset(self, value):
            raise NotImplementedError
        return new_fset

    @classmethod
    def setup_deleted_deleter(cls):
        def new_fdel(self):
            raise NotImplementedError
        return new_fdel


"""
class A(metaclass=LazyMeta):
    @lazy_property
    def a(self) -> int:
        return 0

    @a.setter
    def a(self, value: int) -> None:
        pass

    @lazy_property
    def b(self) -> int:
        return self.a + 1

    @lazy_property
    def c(self) -> int:
        return self.a + 2

    @lazy_property
    def d(self) -> int:
        return self.b + self.c


print(111)
a = A()
print(a.a is None)
a.a = 1
print(a.d)
print(a.__class__._dependency_dict_)
"""
