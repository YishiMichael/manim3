__all__ = [
    "Lazy",
    "LazyMode"
]


from enum import Enum
import inspect
import re
from typing import (
    Any,
    Callable,
    Concatenate,
    #Generic,
    Hashable,
    Literal,
    ParamSpec,
    TypeVar,
    overload
)

from bidict import bidict

from ..lazy.core import (
    LazyCollection,
    LazyCollectionPropertyDescriptor,
    LazyCollectionVariableDescriptor,
    LazyObject,
    LazyObjectPropertyDescriptor,
    LazyObjectVariableDescriptor,
    LazyWrapper
)


_T = TypeVar("_T")
_HashableT = TypeVar("_HashableT", bound=Hashable)
_LazyObjectT = TypeVar("_LazyObjectT", bound="LazyObject")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
_PropertyParameters = ParamSpec("_PropertyParameters")


class AnnotationUtils:
    @classmethod
    def get_return_type(
        cls,
        method: Callable
    ) -> Any:
        if isinstance(return_type := inspect.signature(method).return_annotation, str):
            return NotImplemented
        if isinstance(return_type, type):
            return return_type
        return return_type.__origin__

    @classmethod
    def get_element_return_type(
        cls,
        method: Callable
    ) -> Any:
        if isinstance(collection_type := inspect.signature(method).return_annotation, str):
            return NotImplemented
        assert collection_type.__origin__ is LazyCollection
        return_type = collection_type.__args__[0]
        if isinstance(return_type, type):
            return return_type
        return return_type.__origin__

    @classmethod
    def get_parameter_items(
        cls,
        method: Callable
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[bool, ...]]:
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
        requires_unwrapping_tuple = tuple(
            requires_unwrapping
            for _, requires_unwrapping in parameter_items
        )
        return parameter_name_chains, requires_unwrapping_tuple


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
    __slots__ = ("default_object",)

    def __init__(
        self,
        method: Callable[[type[_InstanceT]], _T]
    ) -> None:
        def new_method(
            cls: type[_InstanceT]
        ) -> LazyWrapper[_T]:
            if (default_object := self.default_object) is None:
                default_object = LazyWrapper(method(cls))
                self.default_object = default_object
            return default_object

        self.default_object: LazyWrapper[_T] | None = None
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
    __slots__ = (
        "content_to_object_bidict",
        "default_object"
    )

    def __init__(
        self,
        method: Callable[[type[_InstanceT]], _HashableT]
    ) -> None:
        def new_method(
            cls: type[_InstanceT]
        ) -> LazyWrapper[_HashableT]:
            if (default_object := self.default_object) is None:
                default_object = LazyWrapper(method(cls))
                self.default_object = default_object
            return default_object

        self.content_to_object_bidict: bidict[_HashableT, LazyWrapper[_HashableT]] = bidict()
        self.default_object: LazyWrapper[_HashableT] | None = None
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
            #print(self.content_to_object_bidict)
            #print(obj, cached_object)
            self.content_to_object_bidict[obj] = cached_object
        super().__set__(instance, cached_object)

    #def clear_ref(
    #    self,
    #    instance: _InstanceT
    #) -> None:
    #    self.content_to_object_bidict.inverse.pop(self.get(instance))
    #    super().clear_ref(instance)


class LazyObjectPropertyDecorator(LazyObjectPropertyDescriptor[_InstanceT, _LazyObjectT]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[..., _LazyObjectT]
    ) -> None:
        parameter_name_chains, requires_unwrapping_tuple = AnnotationUtils.get_parameter_items(method)
        super().__init__(
            element_type=AnnotationUtils.get_return_type(method),
            method=method,
            parameter_name_chains=parameter_name_chains,
            requires_unwrapping_tuple=requires_unwrapping_tuple
        )


class LazyCollectionPropertyDecorator(LazyCollectionPropertyDescriptor[_InstanceT, _LazyObjectT]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[..., LazyCollection[_LazyObjectT]]
    ) -> None:
        parameter_name_chains, requires_unwrapping_tuple = AnnotationUtils.get_parameter_items(method)
        super().__init__(
            element_type=AnnotationUtils.get_element_return_type(method),
            method=method,
            parameter_name_chains=parameter_name_chains,
            requires_unwrapping_tuple=requires_unwrapping_tuple
        )


class LazyObjectPropertyUnwrappedDecorator(LazyObjectPropertyDescriptor[_InstanceT, LazyWrapper[_T]]):
    __slots__ = ()

    def __init__(
        self,
        method: Callable[..., _T]
    ) -> None:
        def new_method(
            cls: type[_InstanceT],
            *args: Any
        ) -> LazyWrapper[_T]:
            return LazyWrapper(method(cls, *args))

        #self.release_method: Callable[[type[_InstanceT], _T], None] | None = None
        parameter_name_chains, requires_unwrapping_tuple = AnnotationUtils.get_parameter_items(method)
        super().__init__(
            element_type=LazyWrapper,
            method=new_method,
            parameter_name_chains=parameter_name_chains,
            requires_unwrapping_tuple=requires_unwrapping_tuple
        )

    #def release_entity(
    #    self,
    #    instance_type: type[_InstanceT],
    #    entity: LazyWrapper[_T]
    #) -> None:
    #    super().release_entity(instance_type, entity)
    #    if self.release_method is not None:
    #        self.release_method(instance_type, entity.value)

    def releaser(
        self,
        release_method: Any
    ) -> Any:
        assert isinstance(release_method, classmethod)
        func = release_method.__func__

        def new_release_method(
            cls: type[_InstanceT],
            entity: LazyWrapper[_T]
        ) -> None:
            func(cls, entity.value)

        self.release_method = new_release_method
        return release_method


class LazyObjectPropertySharedDecorator(LazyObjectPropertyDescriptor[_InstanceT, LazyWrapper[_HashableT]]):
    __slots__ = ("content_to_object_bidict",)

    def __init__(
        self,
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _HashableT]
    ) -> None:
        def new_method(
            cls: type[_InstanceT],
            *args: _PropertyParameters.args,
            **kwargs: _PropertyParameters.kwargs
        ) -> LazyWrapper[_HashableT]:
            content = method(cls, *args, **kwargs)
            if (cached_object := self.content_to_object_bidict.get(content)) is None:
                cached_object = LazyWrapper(content)
                self.content_to_object_bidict[content] = cached_object
            return cached_object

        #self.restock_method: Callable[[_HashableT], None] | None = None
        self.content_to_object_bidict: bidict[_HashableT, LazyWrapper[_HashableT]] = bidict()
        parameter_name_chains, requires_unwrapping_tuple = AnnotationUtils.get_parameter_items(method)
        super().__init__(
            element_type=LazyWrapper,
            method=new_method,
            parameter_name_chains=parameter_name_chains,
            requires_unwrapping_tuple=requires_unwrapping_tuple
        )

    #def clear_ref(
    #    self,
    #    instance: _InstanceT
    #) -> None:
    #    if (obj := self.get_property(instance)._get()) is not None:
    #        self.content_to_object_bidict.inverse.pop(obj)
    #        if self.restock_method is not None:
    #            self.restock_method(obj.value)
    #    super().clear_ref(instance)

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
    ) -> Callable[
        [Callable[[type[_InstanceT]], _LazyObjectT]],
        LazyObjectVariableDecorator[_InstanceT, _LazyObjectT]
    ]: ...

    @overload
    @classmethod
    def variable(
        cls,
        mode: Literal[LazyMode.COLLECTION]
    ) -> Callable[
        [Callable[[type[_InstanceT]], LazyCollection[_LazyObjectT]]],
        LazyCollectionVariableDecorator[_InstanceT, _LazyObjectT]
    ]: ...

    @overload
    @classmethod
    def variable(
        cls,
        mode: Literal[LazyMode.UNWRAPPED]
    ) -> Callable[
        [Callable[[type[_InstanceT]], _T]],
        LazyObjectVariableUnwrappedDecorator[_InstanceT, _T]
    ]: ...

    @overload
    @classmethod
    def variable(
        cls,
        mode: Literal[LazyMode.SHARED]
    ) -> Callable[
        [Callable[[type[_InstanceT]], _HashableT]],
        LazyObjectVariableSharedDecorator[_InstanceT, _HashableT]
    ]: ...

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
            assert isinstance(cls_method, classmethod)
            return decorator_cls(cls_method.__func__)
        return result

    @overload
    @classmethod
    def property(
        cls,
        mode: Literal[LazyMode.OBJECT]
    ) -> Callable[
        [Callable[Concatenate[type[_InstanceT], _PropertyParameters], _LazyObjectT]],
        LazyObjectPropertyDecorator[_InstanceT, _LazyObjectT]
    ]: ...

    @overload
    @classmethod
    def property(
        cls,
        mode: Literal[LazyMode.COLLECTION]
    ) -> Callable[
        [Callable[Concatenate[type[_InstanceT], _PropertyParameters], LazyCollection[_LazyObjectT]]],
        LazyCollectionPropertyDecorator[_InstanceT, _LazyObjectT]
    ]: ...

    @overload
    @classmethod
    def property(
        cls,
        mode: Literal[LazyMode.UNWRAPPED]
    ) -> Callable[
        [Callable[Concatenate[type[_InstanceT], _PropertyParameters], _T]],
        LazyObjectPropertyUnwrappedDecorator[_InstanceT, _T]
    ]: ...

    @overload
    @classmethod
    def property(
        cls,
        mode: Literal[LazyMode.SHARED]
    ) -> Callable[
        [Callable[Concatenate[type[_InstanceT], _PropertyParameters], _HashableT]],
        LazyObjectPropertySharedDecorator[_InstanceT, _HashableT]
    ]: ...

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
            assert isinstance(cls_method, classmethod)
            return decorator_cls(cls_method.__func__)
        return result
