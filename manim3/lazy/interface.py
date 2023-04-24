#import inspect
#import re
from typing import (
    Any,
    Callable,
    Concatenate,
    Hashable,
    Iterable,
    #Literal,
    ParamSpec,
    TypeVar
    #final,
    #overload
)
#import weakref

from ..lazy.core import (
    #LazyConverter,
    LazyCollectionConverter,
    LazyDynamicContainer,
    LazyExternalConverter,
    LazyIndividualConverter,
    #LazyMode,
    #LazyDynamicPropertyDescriptor,
    #LazyDynamicVariableDescriptor,
    LazyObject,
    LazyPropertyDescriptor,
    LazySharedConverter,
    LazyUnitaryContainer,
    LazyVariableDescriptor,
    #LazyUnitaryPropertyDescriptor,
    #LazyUnitaryVariableDescriptor,
    LazyWrapper
)


_T = TypeVar("_T")
_HT = TypeVar("_HT", bound=Hashable)
_ElementT = TypeVar("_ElementT", bound="LazyObject")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
#_DescriptorSetT = TypeVar("_DescriptorSetT")
_PropertyParameters = ParamSpec("_PropertyParameters")


#class AnnotationUtils:
#    __slots__ = ()

#    def __new__(cls):
#        raise TypeError

#    @classmethod
#    def get_return_type(
#        cls,
#        method: Callable
#    ) -> Any:
#        if isinstance(return_type := inspect.signature(method).return_annotation, str):
#            return NotImplemented
#        if isinstance(return_type, type):
#            return return_type
#        return return_type.__origin__

#    @classmethod
#    def get_element_return_type(
#        cls,
#        method: Callable
#    ) -> Any:
#        if isinstance(collection_type := inspect.signature(method).return_annotation, str):
#            return NotImplemented
#        assert issubclass(collection_type.__origin__, Iterable)
#        return_type = collection_type.__args__[0]
#        if isinstance(return_type, type):
#            return return_type
#        return return_type.__origin__

#    @classmethod
#    def get_parameter_items(
#        cls,
#        method: Callable
#    ) -> tuple[tuple[tuple[str, ...], ...], tuple[bool, ...]]:
#        parameter_items = tuple(
#            (name, False) if re.fullmatch(r"_\w+_", name) else (f"_{name}_", True)
#            for name in tuple(inspect.signature(method).parameters)[1:]  # Remove `cls`.
#        )
#        parameter_name_chains = tuple(
#            tuple(re.findall(r"_\w+?_(?=_|$)", parameter_name))
#            for parameter_name, _ in parameter_items
#        )
#        assert all(
#            "".join(parameter_name_chain) == parameter_name
#            for parameter_name_chain, (parameter_name, _) in zip(parameter_name_chains, parameter_items, strict=True)
#        )
#        requires_unwrapping_tuple = tuple(
#            requires_unwrapping
#            for _, requires_unwrapping in parameter_items
#        )
#        return parameter_name_chains, requires_unwrapping_tuple




#class LazyUnitaryVariableDescriptor(LazyVariableDescriptor[
#    _InstanceT, LazyUnitaryContainer[_ElementT], _ElementT, _ElementT, _DescriptorSetT
#]):
#    __slots__ = ()

#    def convert_get(
#        self,
#        container: LazyUnitaryContainer[_ElementT]
#    ) -> _ElementT:
#        return container._element


#class LazyDynamicVariableDescriptor(LazyVariableDescriptor[
#    _InstanceT, LazyDynamicContainer[_ElementT], _ElementT, LazyDynamicContainer[_ElementT], _DescriptorSetT
#]):
#    __slots__ = ()

#    def convert_get(
#        self,
#        container: LazyDynamicContainer[_ElementT]
#    ) -> LazyDynamicContainer[_ElementT]:
#        return container


class LazyVariableIndividualDecorator(LazyVariableDescriptor[
    _InstanceT, LazyUnitaryContainer[_ElementT], _ElementT, _ElementT, _ElementT
]):
    __slots__ = ()

    _converter_class: type[LazyIndividualConverter[_ElementT]] = LazyIndividualConverter

    #def __init__(
    #    self,
    #    method: Callable[[type[_InstanceT]], _ElementT]
    #) -> None:
    #    super().__init__(method)
    #    self.converter = LazyIndividualConverter()


class LazyVariableCollectionDecorator(LazyVariableDescriptor[
    _InstanceT, LazyDynamicContainer[_ElementT], _ElementT, LazyDynamicContainer[_ElementT], Iterable[_ElementT]
]):
    __slots__ = ()

    _converter_class: type[LazyCollectionConverter[_ElementT]] = LazyCollectionConverter

    #def __init__(
    #    self,
    #    method: Callable[[type[_InstanceT]], list[_ElementT]]
    #) -> None:
    #    super().__init__(method)
    #    self.converter = LazyCollectionConverter()


class LazyVariableExternalDecorator(LazyVariableDescriptor[
    _InstanceT, LazyUnitaryContainer[LazyWrapper[_T]], LazyWrapper[_T], LazyWrapper[_T], _T | LazyWrapper[_T]
]):
    __slots__ = ()

    _converter_class: type[LazyExternalConverter[_T]] = LazyExternalConverter

    #def __init__(
    #    self,
    #    method: Callable[[type[_InstanceT]], _T]
    #) -> None:
    #    super().__init__(method)
    #    self.converter = LazyExternalConverter()


class LazyVariableSharedDecorator(LazyVariableDescriptor[
    _InstanceT, LazyUnitaryContainer[LazyWrapper[_HT]], LazyWrapper[_HT], LazyWrapper[_HT], _HT | LazyWrapper[_HT]
]):
    __slots__ = ()

    _converter_class: type[LazyExternalConverter[_HT]] = LazySharedConverter

    #def __init__(
    #    self,
    #    method: Callable[[type[_InstanceT]], _HT]
    #) -> None:
    #    super().__init__(method)
    #    self.converter = LazySharedConverter()


class LazyPropertyIndividualDecorator(LazyPropertyDescriptor[
    _InstanceT, LazyUnitaryContainer[_ElementT], _ElementT, _ElementT, _ElementT
]):
    __slots__ = ()

    _converter_class: type[LazyIndividualConverter[_ElementT]] = LazyIndividualConverter

    #def __init__(
    #    self,
    #    method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _ElementT]
    #) -> None:
    #    super().__init__(method)
    #    self.converter = LazyIndividualConverter()


class LazyPropertyCollectionDecorator(LazyPropertyDescriptor[
    _InstanceT, LazyDynamicContainer[_ElementT], _ElementT, LazyDynamicContainer[_ElementT], Iterable[_ElementT]
]):
    __slots__ = ()

    _converter_class: type[LazyCollectionConverter[_ElementT]] = LazyCollectionConverter

    #def __init__(
    #    self,
    #    method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], list[_ElementT]]
    #) -> None:
    #    super().__init__(method)
    #    self.converter = LazyCollectionConverter()


class LazyPropertyExternalDecorator(LazyPropertyDescriptor[
    _InstanceT, LazyUnitaryContainer[LazyWrapper[_T]], LazyWrapper[_T], LazyWrapper[_T], _T | LazyWrapper[_T]
]):
    __slots__ = ()

    _converter_class: type[LazyExternalConverter[_T]] = LazyExternalConverter

    #def __init__(
    #    self,
    #    method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _T]
    #) -> None:
    #    super().__init__(method)
    #    self.converter = LazyExternalConverter()

    def finalizer(
        self,
        finalize_method: Any
    ) -> Any:
        assert isinstance(finalize_method, classmethod)
        func = finalize_method.__func__

        def new_finalize_method(
            cls: type[_InstanceT],
            value: LazyWrapper[_T]
        ) -> None:
            func(cls, value.value)

        self.finalize_method = new_finalize_method
        return finalize_method


class LazyPropertySharedDecorator(LazyPropertyDescriptor[
    _InstanceT, LazyUnitaryContainer[LazyWrapper[_HT]], LazyWrapper[_HT], LazyWrapper[_HT], _HT | LazyWrapper[_HT]
]):
    __slots__ = ()

    _converter_class: type[LazyExternalConverter[_HT]] = LazySharedConverter

    #def __init__(
    #    self,
    #    method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _HT]
    #) -> None:
    #    super().__init__(method)
    #    self.converter = LazySharedConverter()


#class LazyUnitaryPropertyDescriptor(LazyPropertyDescriptor[
#    _InstanceT, LazyUnitaryContainer[_ElementT], _ElementT, _ElementT, _DescriptorSetT
#]):
#    __slots__ = ()

#    def convert_get(
#        self,
#        container: LazyUnitaryContainer[_ElementT]
#    ) -> _ElementT:
#        return container._element


#class LazyDynamicPropertyDescriptor(LazyPropertyDescriptor[
#    _InstanceT, LazyDynamicContainer[_ElementT], _ElementT, LazyDynamicContainer[_ElementT], _DescriptorSetT
#]):
#    __slots__ = ()

#    def convert_get(
#        self,
#        container: LazyDynamicContainer[_ElementT]
#    ) -> LazyDynamicContainer[_ElementT]:
#        return container


#class LazyUnitaryPropertyDecorator(LazyUnitaryPropertyDescriptor[_InstanceT, _ElementT, _ElementT]):
#    __slots__ = ()

#    def __init__(
#        self,
#        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _ElementT]
#    ) -> None:
#        #parameter_name_chains, requires_unwrapping_tuple = AnnotationUtils.get_parameter_items(method)
#        super().__init__(method)

#    def convert_set(
#        self,
#        new_value: _ElementT
#    ) -> LazyUnitaryContainer[_ElementT]:
#        return LazyUnitaryContainer(
#            element=new_value
#        )


#class LazyDynamicPropertyDecorator(LazyDynamicPropertyDescriptor[_InstanceT, _ElementT, Iterable[_ElementT]]):
#    __slots__ = ()

#    def __init__(
#        self,
#        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], Iterable[_ElementT]]
#    ) -> None:
#        #parameter_name_chains, requires_unwrapping_tuple = AnnotationUtils.get_parameter_items(method)
#        super().__init__(method)

#    def convert_set(
#        self,
#        new_value: Iterable[_ElementT]
#    ) -> LazyDynamicContainer[_ElementT]:
#        return LazyDynamicContainer(
#            elements=new_value
#        )


#class LazyUnitaryPropertyExternalDecorator(LazyUnitaryPropertyDescriptor[_InstanceT, LazyWrapper[_T], _T | LazyWrapper[_T]]):
#    __slots__ = ()

#    def __init__(
#        self,
#        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _T]
#    ) -> None:
#        #parameter_name_chains, requires_unwrapping_tuple = AnnotationUtils.get_parameter_items(method)
#        super().__init__(method)

#    def convert_set(
#        self,
#        new_value: _T | LazyWrapper[_T]
#    ) -> LazyUnitaryContainer[LazyWrapper[_T]]:
#        if not isinstance(new_value, LazyWrapper):
#            new_value = LazyWrapper(new_value)
#        return LazyUnitaryContainer(
#            element=new_value
#        )

#    def finalizer(
#        self,
#        finalize_method: Any
#    ) -> Any:
#        assert isinstance(finalize_method, classmethod)
#        func = finalize_method.__func__

#        def new_finalize_method(
#            cls: type[_InstanceT],
#            value: LazyWrapper[_T]
#        ) -> None:
#            func(cls, value.value)

#        self.finalize_method = new_finalize_method
#        return finalize_method


#class LazyUnitaryPropertySharedDecorator(LazyUnitaryPropertyExternalDecorator[_InstanceT, _HT]):
#    __slots__ = ("content_to_element_dict",)

#    def __init__(
#        self,
#        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _HT]
#    ) -> None:
#        self.content_to_element_dict: weakref.WeakValueDictionary[_HT, LazyWrapper[_HT]] = weakref.WeakValueDictionary()
#        #parameter_name_chains, requires_unwrapping_tuple = AnnotationUtils.get_parameter_items(method)
#        super().__init__(method)

#    def convert_set(
#        self,
#        new_value: _HT | LazyWrapper[_HT]
#    ) -> LazyUnitaryContainer[LazyWrapper[_HT]]:
#        if not isinstance(new_value, LazyWrapper) and (cached_value := self.content_to_element_dict.get(new_value)) is None:
#            cached_value = LazyWrapper(new_value)
#            self.content_to_element_dict[new_value] = cached_value
#            new_value = cached_value
#        return super().convert_set(new_value)


class Lazy:
    __slots__ = ()

    def __new__(cls) -> None:
        raise TypeError

    #@overload
    @classmethod
    def variable(
        cls,
        method: Callable[[type[_InstanceT]], _ElementT]
    ) -> LazyVariableIndividualDecorator[_InstanceT, _ElementT]:
        return LazyVariableIndividualDecorator(method.__func__)

    #@overload
    @classmethod
    def variable_collection(
        cls,
        method: Callable[[type[_InstanceT]], list[_ElementT]]
    ) -> LazyVariableCollectionDecorator[_InstanceT, _ElementT]:
        return LazyVariableCollectionDecorator(method.__func__)

    #@overload
    @classmethod
    def variable_external(
        cls,
        method: Callable[[type[_InstanceT]], _T]
    ) -> LazyVariableExternalDecorator[_InstanceT, _T]:
        return LazyVariableExternalDecorator(method.__func__)

    #@overload
    @classmethod
    def variable_shared(
        cls,
        method: Callable[[type[_InstanceT]], _HT]
    ) -> LazyVariableSharedDecorator[_InstanceT, _HT]:
        return LazyVariableSharedDecorator(method.__func__)

    #@classmethod
    #def variable(
    #    cls,
    #    mode: LazyMode
    #) -> Callable[[Callable], Any]:
    #    if mode is LazyMode.OBJECT:
    #        decorator_cls = LazyVariableIndividualDecorator
    #    elif mode is LazyMode.UNWRAPPED:
    #        decorator_cls = LazyUnitaryVariableUnwrappedDecorator
    #    elif mode is LazyMode.SHARED:
    #        decorator_cls = LazyUnitaryVariableSharedDecorator
    #    elif mode is LazyMode.COLLECTION:
    #        decorator_cls = LazyDynamicVariableDecorator
    #    else:
    #        raise ValueError

    #    def result(
    #        cls_method: Callable
    #    ) -> Any:
    #        assert isinstance(cls_method, classmethod)
    #        return decorator_cls(cls_method.__func__)

    #    return result

    #@overload
    @classmethod
    def property(
        cls,
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _ElementT]
    ) -> LazyPropertyIndividualDecorator[_InstanceT, _ElementT]:
        return LazyPropertyIndividualDecorator(method.__func__)

    #@overload
    @classmethod
    def property_collection(
        cls,
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], list[_ElementT]]
    ) -> LazyPropertyCollectionDecorator[_InstanceT, _ElementT]:
        return LazyPropertyCollectionDecorator(method.__func__)

    #@overload
    @classmethod
    def property_external(
        cls,
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _T]
    ) -> LazyPropertyExternalDecorator[_InstanceT, _T]:
        return LazyPropertyExternalDecorator(method.__func__)

    #@overload
    @classmethod
    def property_shared(
        cls,
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _HT]
    ) -> LazyPropertySharedDecorator[_InstanceT, _HT]:
        return LazyPropertySharedDecorator(method.__func__)

    #@classmethod
    #def property(
    #    cls,
    #    mode: LazyMode
    #) -> Callable[[Callable], Any]:
    #    if mode is LazyMode.OBJECT:
    #        decorator_cls = LazyUnitaryPropertyDecorator
    #    elif mode is LazyMode.UNWRAPPED:
    #        decorator_cls = LazyUnitaryPropertyUnwrappedDecorator
    #    elif mode is LazyMode.SHARED:
    #        decorator_cls = LazyUnitaryPropertySharedDecorator
    #    elif mode is LazyMode.COLLECTION:
    #        decorator_cls = LazyDynamicPropertyDecorator
    #    else:
    #        raise ValueError

    #    def result(
    #        cls_method: Callable
    #    ) -> Any:
    #        assert isinstance(cls_method, classmethod)
    #        return decorator_cls(cls_method.__func__)

    #    return result
