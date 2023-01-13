__all__ = [
    "lazy_property",
    "lazy_property_initializer",
    "lazy_property_initializer_writable",
    "LazyBase"
]


from abc import ABC
import inspect
from types import (
    GenericAlias,
    UnionType
)
from typing import (
    Any,
    Callable,
    ClassVar,
    Concatenate,
    Generic,
    ParamSpec,
    TypeVar,
    overload
)

from ..utils.node import Node


_T = TypeVar("_T")
_R = TypeVar("_R")
_P = ParamSpec("_P")
_Annotation = Any
_LazyBaseT = TypeVar("_LazyBaseT", bound="LazyBase")


class lazy_property(Generic[_LazyBaseT, _T], Node):
    def __init__(self, static_method: Callable[..., _T]):
        #assert isinstance(method, staticmethod)
        method = static_method.__func__
        self.method: Callable[..., _T] = method
        signature = inspect.signature(method)
        self.name: str = method.__name__
        self.annotation: _Annotation = signature.return_annotation
        self.parameters: dict[str, _Annotation] = {
            f"_{parameter.name}_": parameter.annotation
            for parameter in list(signature.parameters.values())
        }
        self.value_dict: dict[_LazyBaseT, _T] = {}
        self.requires_update: dict[_LazyBaseT, bool] = {}
        self.release_method: Callable[[_T], None] | None = None
        self.ancestors: list[lazy_property[_LazyBaseT, _T]] = []
        super().__init__()

    @overload
    def __get__(self, instance: None, owner: type[_LazyBaseT] | None = None) -> "lazy_property[_LazyBaseT, _T]": ...

    @overload
    def __get__(self, instance: _LazyBaseT, owner: type[_LazyBaseT] | None = None) -> _T: ...

    def __get__(self, instance: _LazyBaseT | None, owner: type[_LazyBaseT] | None = None) -> "lazy_property[_LazyBaseT, _T] | _T":
        if instance is None:
            return self
        if not self.requires_update[instance]:
            return self.value_dict[instance]
        if self.release_method is not None:
            if instance in self.value_dict:
                self.release_method(self.value_dict[instance])
        value = self.method(*(
            instance.__getattribute__(parameter)
            for parameter in self.parameters
        ))
        self.value_dict[instance] = value
        self.requires_update[instance] = False
        return value

    @property
    def stripped_name(self) -> str:
        return self.name.strip("_")

    def releaser(self, release_method: Callable[[_T], None]) -> Callable[[_T], None]:
        self.release_method = release_method
        return release_method

    def add_instance(self, instance: _LazyBaseT) -> None:
        self.requires_update[instance] = True

    def update_ancestors_cache(self) -> None:
        self.ancestors = self.get_ancestors()

    def expire_instance(self, instance: _LazyBaseT) -> None:
        for expired_prop in self.ancestors:
            expired_prop.requires_update[instance] = True


class lazy_property_initializer(lazy_property[_LazyBaseT, _T]):
    @overload
    def __get__(self, instance: None, owner: type[_LazyBaseT] | None = None) -> "lazy_property_initializer[_LazyBaseT, _T]": ...

    @overload
    def __get__(self, instance: _LazyBaseT, owner: type[_LazyBaseT] | None = None) -> _T: ...

    def __get__(self, instance: _LazyBaseT | None, owner: type[_LazyBaseT] | None = None) -> "lazy_property_initializer[_LazyBaseT, _T] | _T":
        if instance is None:
            return self
        return self.value_dict[instance]

    def __set__(self, instance: _LazyBaseT, value: _T) -> None:
        raise ValueError("Attempting to set a readonly property")

    def add_instance(self, instance: _LazyBaseT) -> None:
        self.value_dict[instance] = self.method()

    def updater(self, update_method: Callable[Concatenate[_LazyBaseT, _P], _R]) -> Callable[Concatenate[_LazyBaseT, _P], _R]:
        def new_update_method(instance: _LazyBaseT, *args: _P.args, **kwargs: _P.kwargs) -> _R:
            self.expire_instance(instance)
            return update_method(instance, *args, **kwargs)
        return new_update_method


class lazy_property_initializer_writable(lazy_property_initializer[_LazyBaseT, _T]):
    def __set__(self, instance: _LazyBaseT, value: _T) -> None:
        self.expire_instance(instance)
        self.value_dict[instance] = value


class LazyBase(ABC):
    _PROPERTIES: ClassVar[list[lazy_property]]

    def __init_subclass__(cls) -> None:
        properties: dict[str, lazy_property] = {}
        for parent_cls in cls.__mro__[::-1]:
            for name, method in parent_cls.__dict__.items():
                if name not in properties:
                    if isinstance(method, lazy_property):
                        properties[name] = method
                    continue
                assert isinstance(method, lazy_property)
                cls._check_annotation_matching(method.annotation, properties[name].annotation)
                properties[name] = method

        for prop in properties.values():
            if isinstance(prop, lazy_property_initializer):
                assert not prop.parameters
                continue
            for param_name, param_annotation in prop.parameters.items():
                cls._check_annotation_matching(properties[param_name].annotation, param_annotation)
                prop.add(properties[param_name])
        for prop in properties.values():
            prop.update_ancestors_cache()

        cls._PROPERTIES = list(properties.values())
        return super().__init_subclass__()

    def __init__(self) -> None:
        for prop in self._PROPERTIES:
            prop.add_instance(self)
        super().__init__()

    @classmethod
    def _check_annotation_matching(cls, child_annotation: _Annotation, parent_annotation: _Annotation) -> None:
        def _to_classes(annotation: _Annotation) -> tuple[type, ...]:
            return tuple(
                child.__origin__ if isinstance(child, GenericAlias) else
                Callable if isinstance(child, Callable) else child
                for child in (
                    annotation.__args__ if isinstance(annotation, UnionType) else (annotation,)
                )
            )

        assert all(
            any(
                issubclass(child_cls, parent_cls)
                for parent_cls in _to_classes(parent_annotation)
            )
            for child_cls in _to_classes(child_annotation)
        ), f"Type annotation mismatched: `{child_annotation}` is not compatible with `{parent_annotation}`"


"""
class A(LazyBase):
    @lazy_property
    @staticmethod
    def _p_(q: str) -> int:
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
