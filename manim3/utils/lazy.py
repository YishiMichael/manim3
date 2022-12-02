__all__ = [
    "lazy_property",
    "lazy_property_initializer",
    "lazy_property_initializer_writable",
    "LazyBase"
]


from abc import ABC
from typing import Callable, ClassVar, Concatenate, Generic, ParamSpec, Type, TypeVar, overload


_T = TypeVar("_T")
_R = TypeVar("_R")
_P = ParamSpec("_P")
_LazyBaseT = TypeVar("_LazyBaseT", bound="LazyBase")


class lazy_property(Generic[_T]):
    def __init__(self, class_method: Callable[..., _T]):
        assert isinstance(class_method, classmethod)
        self.class_method: Callable[..., _T] = class_method.__func__
        code = class_method.__func__.__code__
        self.varnames: list[str] = [
            f"_{varname}_"
            for varname in code.co_varnames[1:code.co_argcount]  # ignore cls
        ]
        self.values: dict[LazyBase, _T] = {}
        self.requires_update: dict[LazyBase, bool] = {}

    @overload
    def __get__(self, instance: None, owner: "Type[LazyBase] | None" = None) -> "lazy_property": ...

    @overload
    def __get__(self, instance: "LazyBase", owner: "Type[LazyBase] | None" = None) -> _T: ...

    def __get__(self, instance: "LazyBase | None", owner: "Type[LazyBase] | None" = None) -> "lazy_property | _T":
        if instance is None:
            return self
        if not self.requires_update[instance]:
            return self.values[instance]
        value = self.class_method(owner, *(
            instance.__getattribute__(varname)
            for varname in self.varnames
        ))
        self.values[instance] = value
        self.requires_update[instance] = False
        return value

    def add_instance(self, instance: "LazyBase") -> None:
        self.requires_update[instance] = True


class lazy_property_initializer(Generic[_T]):
    def __init__(self, static_method: Callable[[], _T]):
        assert isinstance(static_method, staticmethod)
        self.static_method: Callable[[], _T] = static_method.__func__
        self.values: dict[LazyBase, _T] = {}

    @overload
    def __get__(self, instance: None, owner: "Type[LazyBase] | None" = None) -> "lazy_property_initializer": ...

    @overload
    def __get__(self, instance: "LazyBase", owner: "Type[LazyBase] | None" = None) -> _T: ...

    def __get__(self, instance: "LazyBase | None", owner: "Type[LazyBase] | None" = None) -> "lazy_property_initializer | _T":
        if instance is None:
            return self
        return self.values[instance]

    def __set__(self, instance: "LazyBase", value: _T) -> None:
        raise ValueError("Attempting to set a readonly property")

    def add_instance(self, instance: "LazyBase") -> None:
        self.values[instance] = self.static_method()

    def updater(self, update_method: Callable[Concatenate[_LazyBaseT, _P], _R]) -> Callable[Concatenate[_LazyBaseT, _P], _R]:
        def new_update_method(instance: _LazyBaseT, *args: _P.args, **kwargs: _P.kwargs) -> _R:
            instance._expire(self)
            return update_method(instance, *args, **kwargs)
        return new_update_method


class lazy_property_initializer_writable(lazy_property_initializer[_T]):
    def __set__(self, instance: "LazyBase", value: _T) -> None:
        instance._expire(self)
        self.values[instance] = value


class LazyBase(ABC):
    _INITIALIZERS: ClassVar[set[lazy_property_initializer]] = set()
    _PROPERTIES: ClassVar[set[lazy_property]] = set()
    _EXPIRE_DICT: ClassVar[dict[lazy_property_initializer, set[lazy_property]]] = {}

    def __init_subclass__(cls) -> None:
        initializers: dict[str, lazy_property_initializer] = {}
        properties: dict[str, lazy_property] = {}
        expire_dict: dict[lazy_property_initializer, set[lazy_property]] = {}
        parameters_dict: dict[str, list[str]] = {}

        mathods = {}
        for parent_cls in cls.__mro__[::-1]:
            mathods.update(parent_cls.__dict__)
        for name, method in mathods.items():
            #if not name.startswith("_") and name.endswith("_"):
            #    continue
            if isinstance(method, lazy_property_initializer):
                initializers[name] = method
            elif isinstance(method, lazy_property):
                properties[name] = method
                parameters_dict[name] = method.varnames

        for name, initializer in initializers.items():
            expired = set()
            extended = {
                param for param, args in parameters_dict.items()
                if name in args
            }
            while expired != extended:
                expired = extended
                extended = expired.union({
                    param for param, args in parameters_dict.items()
                    if any(arg in args for arg in expired)
                })
            expire_dict[initializer] = {
                properties[property_name]
                for property_name in expired
            }

        cls._INITIALIZERS = set(initializers.values())
        cls._PROPERTIES = set(properties.values())
        cls._EXPIRE_DICT = expire_dict

    def __init__(self) -> None:
        for initializer in self._INITIALIZERS:
            initializer.add_instance(self)
        for prop in self._PROPERTIES:
            prop.add_instance(self)
        super().__init__()

    def _expire(self, initializer: lazy_property_initializer) -> None:
        for expired_prop in self._EXPIRE_DICT[initializer]:
            expired_prop.requires_update[self] = True


"""
class A(LazyBase):
    @lazy_property
    @classmethod
    def _p_(cls, q: str):
        return int(q)

    @lazy_property_initializer_writable
    @staticmethod
    def _q_() -> str:
        return "2"


class B(A):
    pass


a = B()
s = a._p_ + 3
a._q_ += "8"
print(s, a._p_)
"""
