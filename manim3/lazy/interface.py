from typing import (
    Any,
    Callable,
    Concatenate,
    Hashable,
    Iterable,
    ParamSpec,
    TypeVar
)

from ..lazy.core import (
    LazyCollectionConverter,
    LazyDynamicContainer,
    LazyExternalConverter,
    LazyIndividualConverter,
    LazyObject,
    LazyPropertyDescriptor,
    LazySharedConverter,
    LazyUnitaryContainer,
    LazyVariableDescriptor,
    LazyWrapper
)


_T = TypeVar("_T")
_HT = TypeVar("_HT", bound=Hashable)
_ElementT = TypeVar("_ElementT", bound="LazyObject")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
_PropertyParameters = ParamSpec("_PropertyParameters")


class LazyVariableIndividualDecorator(LazyVariableDescriptor[
    _InstanceT, LazyUnitaryContainer[_ElementT], _ElementT, _ElementT, _ElementT
]):
    __slots__ = ()

    _converter_class: type[LazyIndividualConverter[_ElementT]] = LazyIndividualConverter


class LazyVariableCollectionDecorator(LazyVariableDescriptor[
    _InstanceT, LazyDynamicContainer[_ElementT], _ElementT, LazyDynamicContainer[_ElementT], Iterable[_ElementT]
]):
    __slots__ = ()

    _converter_class: type[LazyCollectionConverter[_ElementT]] = LazyCollectionConverter


class LazyVariableExternalDecorator(LazyVariableDescriptor[
    _InstanceT, LazyUnitaryContainer[LazyWrapper[_T]], LazyWrapper[_T], LazyWrapper[_T], _T | LazyWrapper[_T]
]):
    __slots__ = ()

    _converter_class: type[LazyExternalConverter[_T]] = LazyExternalConverter


class LazyVariableSharedDecorator(LazyVariableDescriptor[
    _InstanceT, LazyUnitaryContainer[LazyWrapper[_HT]], LazyWrapper[_HT], LazyWrapper[_HT], _HT | LazyWrapper[_HT]
]):
    __slots__ = ()

    _converter_class: type[LazyExternalConverter[_HT]] = LazySharedConverter


class LazyPropertyIndividualDecorator(LazyPropertyDescriptor[
    _InstanceT, LazyUnitaryContainer[_ElementT], _ElementT, _ElementT, _ElementT
]):
    __slots__ = ()

    _converter_class: type[LazyIndividualConverter[_ElementT]] = LazyIndividualConverter


class LazyPropertyCollectionDecorator(LazyPropertyDescriptor[
    _InstanceT, LazyDynamicContainer[_ElementT], _ElementT, LazyDynamicContainer[_ElementT], Iterable[_ElementT]
]):
    __slots__ = ()

    _converter_class: type[LazyCollectionConverter[_ElementT]] = LazyCollectionConverter


class LazyPropertyExternalDecorator(LazyPropertyDescriptor[
    _InstanceT, LazyUnitaryContainer[LazyWrapper[_T]], LazyWrapper[_T], LazyWrapper[_T], _T | LazyWrapper[_T]
]):
    __slots__ = ()

    _converter_class: type[LazyExternalConverter[_T]] = LazyExternalConverter

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


class Lazy:
    __slots__ = ()

    def __new__(cls) -> None:
        raise TypeError

    @classmethod
    def variable(
        cls,
        method: Callable[[type[_InstanceT]], _ElementT]
    ) -> LazyVariableIndividualDecorator[_InstanceT, _ElementT]:
        return LazyVariableIndividualDecorator(method.__func__)

    @classmethod
    def variable_collection(
        cls,
        method: Callable[[type[_InstanceT]], list[_ElementT]]
    ) -> LazyVariableCollectionDecorator[_InstanceT, _ElementT]:
        return LazyVariableCollectionDecorator(method.__func__)

    @classmethod
    def variable_external(
        cls,
        method: Callable[[type[_InstanceT]], _T]
    ) -> LazyVariableExternalDecorator[_InstanceT, _T]:
        return LazyVariableExternalDecorator(method.__func__)

    @classmethod
    def variable_shared(
        cls,
        method: Callable[[type[_InstanceT]], _HT]
    ) -> LazyVariableSharedDecorator[_InstanceT, _HT]:
        return LazyVariableSharedDecorator(method.__func__)

    @classmethod
    def property(
        cls,
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _ElementT]
    ) -> LazyPropertyIndividualDecorator[_InstanceT, _ElementT]:
        return LazyPropertyIndividualDecorator(method.__func__)

    @classmethod
    def property_collection(
        cls,
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], list[_ElementT]]
    ) -> LazyPropertyCollectionDecorator[_InstanceT, _ElementT]:
        return LazyPropertyCollectionDecorator(method.__func__)

    @classmethod
    def property_external(
        cls,
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _T]
    ) -> LazyPropertyExternalDecorator[_InstanceT, _T]:
        return LazyPropertyExternalDecorator(method.__func__)

    @classmethod
    def property_shared(
        cls,
        method: Callable[Concatenate[type[_InstanceT], _PropertyParameters], _HT]
    ) -> LazyPropertySharedDecorator[_InstanceT, _HT]:
        return LazyPropertySharedDecorator(method.__func__)
