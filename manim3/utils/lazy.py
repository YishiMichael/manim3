__all__ = [
    "lazy_property",
    "lazy_property_initializer",
    "lazy_property_initializer_writable",
    "LazyBase"
]


from abc import ABC
import inspect
from typing import Any, Callable, ClassVar, Concatenate, Generic, ParamSpec, TypeVar, overload

from ..utils.node import Node


_T = TypeVar("_T")
_R = TypeVar("_R")
_P = ParamSpec("_P")
_Annotation = Any
_LazyBaseT = TypeVar("_LazyBaseT", bound="LazyBase")


#class ParameterNode(Node):
#    def __init__(self, name: str, annotation: _Annotation):
#        self.name: str = name
#        self.annotation: _Annotation = annotation
#        super().__init__()


class lazy_property(Node, Generic[_LazyBaseT, _T]):
    def __init__(self, static_method: Callable[..., _T]):
        #assert isinstance(method, staticmethod)
        method = static_method.__func__
        self.method: Callable[..., _T] = method
        signature = inspect.signature(method)
        #node = ParameterNode(method.__name__, signature.return_annotation)
        #node.set_children(
        #    ParameterNode(f"_{parameter.name}_", parameter.annotation)
        #    for parameter in list(signature.parameters.values())[1:]  # ignores cls
        #)
        #self.parameter_node: ParameterNode = node
        self.name: str = method.__name__
        self.annotation: _Annotation = signature.return_annotation
        #self.node: ParameterNode = ParameterNode(method.__name__, signature.return_annotation)
        self.parameters: dict[str, _Annotation] = {
            f"_{parameter.name}_": parameter.annotation
            for parameter in list(signature.parameters.values())
        }
        self.value_dict: dict[_LazyBaseT, _T] = {}
        self.requires_update: dict[_LazyBaseT, bool] = {}
        self.release_method: Callable[[_T], None] | None = None
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

    def add_instance(self, instance: _LazyBaseT) -> None:
        self.requires_update[instance] = True

    def releaser(self, release_method: Callable[[_T], None]) -> Callable[[_T], None]:
        self.release_method = release_method
        return release_method

    def _expire_instance(self, instance: _LazyBaseT) -> None:
        #for expired_prop in self._EXPIRE_DICT[initializer]:
        for expired_prop in self.get_ancestors():
            expired_prop.requires_update[instance] = True


class lazy_property_initializer(lazy_property[_LazyBaseT, _T]):
    #def __init__(self, method: Callable[[type[_LazyBaseT]], _T]):
    #    assert isinstance(method, classmethod)
    #    self.method: Callable[[type[_LazyBaseT]], _T] = method.__func__
    #    self.values: dict[LazyBase, _T] = {}

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
            self._expire_instance(instance)
            return update_method(instance, *args, **kwargs)
        return new_update_method


class lazy_property_initializer_writable(lazy_property_initializer[_LazyBaseT, _T]):
    def __set__(self, instance: _LazyBaseT, value: _T) -> None:
        self._expire_instance(instance)
        self.value_dict[instance] = value


class LazyBase(ABC):
    #_INITIALIZERS: ClassVar[dict[str, lazy_property_initializer]] = {}
    _PROPERTIES: ClassVar[list[lazy_property]]
    #_EXPIRE_DICT: ClassVar[dict[lazy_property, set[lazy_property]]] = {}

    def __init_subclass__(cls) -> None:
        #initializers: dict[str, lazy_property_initializer] = {}

        methods = {}
        for parent_cls in cls.__mro__[::-1]:
            methods.update(parent_cls.__dict__)  # type: ignore

        properties: dict[str, lazy_property] = {
            name: method
            for name, method in methods.items()
            if isinstance(method, lazy_property)
        }
        for prop in properties.values():
            if isinstance(prop, lazy_property_initializer):
                assert not prop.parameters
                continue
            for param_name, param_annotation in prop.parameters.items():
                param_node = properties[param_name]
                # TODO: use issubclass() instead
                assert param_node.annotation == param_annotation, \
                    AssertionError(f"Type annotation mismatched: {param_node.annotation} and {param_annotation}")
                prop.add(param_node)

        #expire_dict: dict[lazy_property_initializer, set[lazy_property]] = {}

        #assemble_nodes([
        #    method.node
        #    for method in properties.values()
        #])

        #for name, initializer in initializers.items():
        #    expired = set()
        #    extended = {
        #        param for param, args in parameters_dict.items()
        #        if name in args
        #    }
        #    while expired != extended:
        #        expired = extended
        #        extended = expired.union({
        #            param for param, args in parameters_dict.items()
        #            if any(arg in args for arg in expired)
        #        })
        #    expire_dict[initializer] = {
        #        properties[property_name]
        #        for property_name in expired
        #    }

        #cls._INITIALIZERS = initializers
        cls._PROPERTIES = list(properties.values())
        #cls._EXPIRE_DICT = {
        #    prop:
        #}
        return super().__init_subclass__()

    def __init__(self) -> None:
        #for initializer in self._INITIALIZERS.values():
        #    initializer.add_instance(self)
        for prop in self._PROPERTIES:
            prop.add_instance(self)
        super().__init__()

    #def _is_expired(self, name: str) -> bool:
    #    return self._PROPERTIES[name].requires_update[self]


"""
class A(LazyBase):
    @lazy_property
    @staticmethod
    def _p_(q: str):
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
