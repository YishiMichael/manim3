__all__ = [
    "Lazy",
    "LazyMode",
    "LazyWrapper"
    #"lazyproperty",
    #"lazyvariable"
    #"lazy_object",
    #"lazy_object_collection",
    #"lazy_object_shared",
    #"lazy_object_unwrapped",
    #"lazy_property",
    #"lazy_property_shared",
    #"lazy_property_unwrapped"
]


from enum import Enum
import inspect
import re
from typing import (
    _GenericAlias,  # TODO
    Any,
    Callable,
    Generic,
    Hashable,
    Literal,
    TypeVar,
    overload
)

from bidict import bidict

from ..lazy.core import (
    #LazyCollection,
    #LazyEntity,
    #LazyObject,
    #LazyObjectCollectionDescriptor,
    #LazyObjectDescriptor,
    #LazyPropertyDescriptor
    LazyCollection,
    LazyCollectionPropertyDescriptor,
    LazyCollectionVariableDescriptor,
    LazyObject,
    LazyObjectPropertyDescriptor,
    LazyObjectVariableDescriptor
)


_T = TypeVar("_T")
_HashableT = TypeVar("_HashableT", bound=Hashable)
#_LazyEntityT = TypeVar("_LazyEntityT", bound="LazyEntity")
_LazyObjectT = TypeVar("_LazyObjectT", bound="LazyObject")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")


class AnnotationUtils:
    @classmethod
    def get_return_type(
        cls,
        method: Callable
    ) -> Any:
        if isinstance(return_type := inspect.signature(method).return_annotation, str):
            return NotImplemented
        if isinstance(return_type, _GenericAlias):
            return return_type.__origin__
        return return_type

    @classmethod
    def get_element_return_type(
        cls,
        method: Callable
    ) -> Any:
        if isinstance(collection_type := inspect.signature(method).return_annotation, str):
            return NotImplemented
        assert isinstance(collection_type, _GenericAlias)
        assert collection_type.__origin__ is LazyCollection
        if isinstance(return_type := collection_type.__args__[0], _GenericAlias):
            return return_type.__origin__
        return return_type

    @classmethod
    def get_parameter_items(
        cls,
        method: Callable
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[Callable[[Any], Any] | None, ...]]:
        parameter_items = tuple(
            (name, False) if re.fullmatch(r"_\w+_", name) else (f"_{name}_", True)
            for name in tuple(inspect.signature(method).parameters)[1:]  # remove `cls`
        )
        parameter_name_chains = tuple(
            tuple(re.findall(r"_\w+?_(?=_|$)", parameter_name))
            for parameter_name, _ in parameter_items
        )
        assert all(
            "".join(parameter_name_chain) == parameter_name
            for parameter_name_chain, (parameter_name, _) in zip(parameter_name_chains, parameter_items, strict=True)
        )
        parameter_preapplied_methods = tuple(
            (lambda obj: obj.value) if requires_unwrapping else None
            for _, requires_unwrapping in parameter_items
        )
        return parameter_name_chains, parameter_preapplied_methods


class LazyWrapper(Generic[_T], LazyObject):
    __slots__ = ("__value",)

    def __init__(
        self,
        value: _T
    ) -> None:
        super().__init__()
        self.__value: _T = value

    @property
    def value(self) -> _T:
        return self.__value


class LazyObjectVariableDecorator(LazyObjectVariableDescriptor[_InstanceT, _LazyObjectT]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[[type[_InstanceT]], _LazyObjectT]
    ) -> None:
        super().__init__(
            element_type=AnnotationUtils.get_return_type(method),
            method=method
        )


class LazyCollectionVariableDecorator(LazyCollectionVariableDescriptor[_InstanceT, _LazyObjectT]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[[type[_InstanceT]], LazyCollection[_LazyObjectT]]
    ) -> None:
        super().__init__(
            element_type=AnnotationUtils.get_element_return_type(method),
            method=method
        )


class LazyObjectVariableUnwrappedDecorator(LazyObjectVariableDescriptor[_InstanceT, LazyWrapper[_T]]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[[type[_InstanceT]], _T]
    ) -> None:
        def new_method(
            cls: type[_InstanceT]
        ) -> LazyWrapper[_T]:
            return LazyWrapper(method(cls))

        super().__init__(
            element_type=LazyWrapper,
            method=new_method
        )

    def __set__(
        self,
        instance: _InstanceT,
        obj: _T | LazyWrapper[_T]
    ) -> None:
        if not isinstance(obj, LazyWrapper):
            obj = LazyWrapper(obj)
        super().__set__(instance, obj)


class LazyObjectVariableSharedDecorator(LazyObjectVariableDescriptor[_InstanceT, LazyWrapper[_HashableT]]):
    __slots__ = ("content_to_object_bidict",)

    def __init__(
        self,
        method: Callable[[type[_InstanceT]], _HashableT]
    ) -> None:
        self.content_to_object_bidict: bidict[_HashableT, LazyWrapper[_HashableT]] = bidict()

        def new_method(
            cls: type[_InstanceT]
        ) -> LazyWrapper[_HashableT]:
            return LazyWrapper(method(cls))

        super().__init__(
            element_type=LazyWrapper,
            method=new_method
        )

    def __set__(
        self,
        instance: _InstanceT,
        obj: _HashableT
    ) -> None:
        if (cached_object := self.content_to_object_bidict.get(obj)) is None:
            cached_object = LazyWrapper(obj)
            self.content_to_object_bidict[obj] = cached_object
        super().__set__(instance, cached_object)

    def restock(
        self,
        instance: _InstanceT
    ) -> None:
        self.content_to_object_bidict.inverse.pop(self.get_entity(instance))
        super().restock(instance)


class LazyObjectPropertyDecorator(LazyObjectPropertyDescriptor[_InstanceT, _LazyObjectT]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[..., _LazyObjectT]
    ) -> None:
        #new_method, parameter_name_chains = self.wrap_parameters(cls_method.__func__)
        parameter_name_chains, parameter_preapplied_methods = AnnotationUtils.get_parameter_items(method)
        super().__init__(
            element_type=AnnotationUtils.get_return_type(method),
            method=method,
            parameter_name_chains=parameter_name_chains,
            parameter_preapplied_methods=parameter_preapplied_methods
        )

    #@classmethod
    #def wrap_parameters(
    #    cls,
    #    method: Callable[..., _LazyEntityT]
    #) -> tuple[Callable[..., _LazyEntityT], tuple[tuple[str, ...], ...]]:
    #    parameter_items = tuple(
    #        (name, False) if re.fullmatch(r"_\w+_", name) else (f"_{name}_", True)
    #        for name in tuple(inspect.signature(method).parameters)[1:]  # remove `cls`
    #    )
    #    parameter_name_chains = tuple(
    #        tuple(re.findall(r"_\w+?_(?=_|$)", parameter_name))
    #        for parameter_name, _ in parameter_items
    #    )
    #    assert all(
    #        "".join(parameter_name_chain) == parameter_name
    #        for parameter_name_chain, (parameter_name, _) in zip(parameter_name_chains, parameter_items, strict=True)
    #    )
    #    requires_unwrapping_tuple = tuple(requires_unwrapping for _, requires_unwrapping in parameter_items)

    #    def parameters_wrapped_method(
    #        kls: type[_InstanceT],
    #        *args: Any,
    #        **kwargs: Any
    #    ) -> _LazyEntityT:
    #        return method(kls, *(
    #            arg if not requires_unwrapping else cls.apply_at_depth(
    #                lambda obj: obj.value, arg, 
    #            )
    #            for arg, requires_unwrapping in zip(args, requires_unwrapping_tuple, strict=True)
    #        ), **kwargs)
    #    return parameters_wrapped_method, parameter_name_chains


class LazyCollectionPropertyDecorator(LazyCollectionPropertyDescriptor[_InstanceT, _LazyObjectT]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[..., LazyCollection[_LazyObjectT]]
    ) -> None:
        parameter_name_chains, parameter_preapplied_methods = AnnotationUtils.get_parameter_items(method)
        super().__init__(
            element_type=AnnotationUtils.get_element_return_type(method),
            method=method,
            parameter_name_chains=parameter_name_chains,
            parameter_preapplied_methods=parameter_preapplied_methods
        )


class LazyObjectPropertyUnwrappedDecorator(LazyObjectPropertyDescriptor[_InstanceT, LazyWrapper[_T]]):
    __slots__ = ("restock_method",)

    def __init__(
        self,
        method: Callable[..., _T]
    ) -> None:
        self.restock_method: Callable[[_T], None] | None = None
        #parameters_wrapped_method, parameter_name_chains = lazy_property.wrap_parameters(cls_method.__func__)

        def new_method(
            cls: type[_InstanceT],
            *args: Any
        ) -> LazyWrapper[_T]:
            return LazyWrapper(method(cls, *args))

        parameter_name_chains, parameter_preapplied_methods = AnnotationUtils.get_parameter_items(method)
        super().__init__(
            element_type=LazyWrapper,
            method=new_method,
            parameter_name_chains=parameter_name_chains,
            parameter_preapplied_methods=parameter_preapplied_methods
        )

    def restock(
        self,
        instance: _InstanceT
    ) -> None:
        if (obj := self.get_property(instance)._get()) is not None:
            if self.restock_method is not None:
                self.restock_method(obj.value)
        super().restock(instance)

    def restocker(
        self,
        restock_method: Callable[[_T], None]
    ) -> Callable[[_T], None]:
        self.restock_method = restock_method
        return restock_method


class LazyObjectPropertySharedDecorator(LazyObjectPropertyDescriptor[_InstanceT, LazyWrapper[_HashableT]]):
    __slots__ = ("content_to_object_bidict",)

    def __init__(
        self,
        method: Callable[..., _HashableT]
    ) -> None:
        #self.restock_method: Callable[[_HashableT], None] | None = None
        self.content_to_object_bidict: bidict[_HashableT, LazyWrapper[_HashableT]] = bidict()
        #parameters_wrapped_method, parameter_name_chains = lazy_property.wrap_parameters(cls_method.__func__)

        def new_method(
            cls: type[_InstanceT],
            *args: Any
        ) -> LazyWrapper[_HashableT]:
            content = method(cls, *args)
            if (cached_object := self.content_to_object_bidict.get(content)) is None:
                cached_object = LazyWrapper(content)
                self.content_to_object_bidict[content] = cached_object
            return cached_object

        parameter_name_chains, parameter_preapplied_methods = AnnotationUtils.get_parameter_items(method)
        super().__init__(
            element_type=LazyWrapper,
            method=new_method,
            parameter_name_chains=parameter_name_chains,
            parameter_preapplied_methods=parameter_preapplied_methods
        )
        #self.restock_methods: list[Callable[[_HashableT], None]] = []

    #def restock(
    #    self,
    #    instance: _InstanceT
    #) -> None:
    #    if (obj := self.get_property(instance)._get()) is not None:
    #        self.content_to_object_bidict.inverse.pop(obj)
    #        if self.restock_method is not None:
    #            self.restock_method(obj.value)
    #    super().restock(instance)

    #def restocker(
    #    self,
    #    restock_method: Callable[[_HashableT], None]
    #) -> Callable[[_HashableT], None]:
    #    self.restock_method = restock_method
    #    return restock_method


class LazyMode(Enum):
    OBJECT = 0
    COLLECTION = 1
    UNWRAPPED = 2
    SHARED = 3


class Lazy:
    @overload
    @classmethod
    def variable(
        cls,
        mode: Literal[LazyMode.OBJECT]
    ) -> Callable[[Callable], LazyObjectVariableDecorator]: ...

    @overload
    @classmethod
    def variable(
        cls,
        mode: Literal[LazyMode.COLLECTION]
    ) -> Callable[[Callable], LazyCollectionVariableDecorator]: ...

    @overload
    @classmethod
    def variable(
        cls,
        mode: Literal[LazyMode.UNWRAPPED]
    ) -> Callable[[Callable], LazyObjectVariableUnwrappedDecorator]: ...

    @overload
    @classmethod
    def variable(
        cls,
        mode: Literal[LazyMode.SHARED]
    ) -> Callable[[Callable], LazyObjectVariableSharedDecorator]: ...

    @classmethod
    def variable(
        cls,
        mode: LazyMode
    ) -> Callable[[Callable], Any]:
        if mode is LazyMode.OBJECT:
            decorator_cls = LazyObjectVariableDecorator
        elif mode is LazyMode.COLLECTION:
            decorator_cls = LazyCollectionVariableDecorator
        elif mode is LazyMode.UNWRAPPED:
            decorator_cls = LazyObjectVariableUnwrappedDecorator
        elif mode is LazyMode.SHARED:
            decorator_cls = LazyObjectVariableSharedDecorator
        else:
            raise ValueError

        def result(
            cls_method: Callable
        ) -> Any:
            return decorator_cls(cls_method.__func__)
        return result

    @overload
    @classmethod
    def property(
        cls,
        mode: Literal[LazyMode.OBJECT]
    ) -> Callable[[Callable], LazyObjectPropertyDecorator]: ...

    @overload
    @classmethod
    def property(
        cls,
        mode: Literal[LazyMode.COLLECTION]
    ) -> Callable[[Callable], LazyCollectionPropertyDecorator]: ...

    @overload
    @classmethod
    def property(
        cls,
        mode: Literal[LazyMode.UNWRAPPED]
    ) -> Callable[[Callable], LazyObjectPropertyUnwrappedDecorator]: ...

    @overload
    @classmethod
    def property(
        cls,
        mode: Literal[LazyMode.SHARED]
    ) -> Callable[[Callable], LazyObjectPropertySharedDecorator]: ...

    @classmethod
    def property(
        cls,
        mode: LazyMode
    ) -> Callable[[Callable], Any]:
        if mode is LazyMode.OBJECT:
            decorator_cls = LazyObjectPropertyDecorator
        elif mode is LazyMode.COLLECTION:
            decorator_cls = LazyCollectionPropertyDecorator
        elif mode is LazyMode.UNWRAPPED:
            decorator_cls = LazyObjectPropertyUnwrappedDecorator
        elif mode is LazyMode.SHARED:
            decorator_cls = LazyObjectPropertySharedDecorator
        else:
            raise ValueError

        def result(
            cls_method: Callable
        ) -> Any:
            return decorator_cls(cls_method.__func__)
        return result
