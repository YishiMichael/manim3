from __future__ import annotations


from typing import (
    TYPE_CHECKING,
    Callable,
    Hashable,
    Never,
    Self,
    overload
)

import attrs

if TYPE_CHECKING:
    from .lazy_object import LazyObject
    from .lazy_routine import LazyRoutine


@attrs.frozen(kw_only=True)
class LazyDescriptorInfo[T, DataT]:
    method: Callable[..., DataT]
    is_plural: bool
    #decomposer: Callable[[DataT], tuple[T, ...]],
    #composer: Callable[[tuple[T, ...]], DataT],
    is_variable: bool
    hasher: Callable[[T], Hashable]
    freeze: bool
    cache_capacity: int


class LazyDescriptor[T, DataT]:
    __slots__ = (
        "_info",
        #"_method",
        #"_name",
        #"_descriptor_info_chains",
        #"_overloading_chains",
        #"_is_plural",
        #"_decomposer",
        #"_composer",
        #"_is_variable",
        #"_hasher",
        #"_freeze",
        #"_freezer",
        #"_cache",
        #"_cache_capacity",
        "_routine"
        #"_parameter_key_registration",
        #"_element_registration"
        #"_element_type",
        #"_element_type_annotation",
        #"_descriptor_chains"
    )

    def __init__(
        self: Self,
        info: LazyDescriptorInfo[T, DataT]
    ) -> None:
        super().__init__()
        self._info: LazyDescriptorInfo[T, DataT] = info
        #self._method: Callable[..., DataT] = method.__func__
        #self._method: Callable[..., DataT] = method
        #self._name: str = method.__name__
        #self._descriptor_info_chains: tuple[tuple[tuple[str, bool], ...], ...] = NotImplemented
        #self._overloading_chains: tuple[tuple[LazyOverloading, ...], ...] = NotImplemented
        #self._is_plural: bool = is_plural
        #self._decomposer: Callable[[DataT], tuple[T, ...]] = decomposer
        #self._composer: Callable[[tuple[T, ...]], DataT] = composer
        #self._is_variable: bool = is_variable
        #self._hasher: Callable[[T], Hashable] = hasher
        #self._freeze: bool = freeze
        #self._cache_capacity: int = cache_capacity
        #self._cache: Cache[Registered[Hashable], tuple[Registered[T], ...]] = Cache(capacity=cache_capacity)
        #self._parameter_key_registration: Registration[Hashable, Hashable] = Registration()

        #self._element_type: type[T] | None = None
        #self._element_type_annotation: type = NotImplemented
        #self._descriptor_chains: tuple[tuple[LazyDescriptor, ...], ...] = NotImplemented
        self._routine: LazyRoutine[T, DataT] = NotImplemented

    @overload
    def __get__(
        self: Self,
        instance: None,
        owner: type[LazyObject] | None = None
    ) -> Self: ...

    @overload
    def __get__(
        self: Self,
        instance: LazyObject,
        owner: type[LazyObject] | None = None
    ) -> DataT: ...

    def __get__(
        self: Self,
        instance: LazyObject | None,
        owner: type[LazyObject] | None = None
    ) -> Self | DataT:
        if instance is None:
            return self
        return self._routine.descriptor_get(instance)

    def __set__(
        self: Self,
        instance: LazyObject,
        data: DataT
    ) -> None:
        self._routine.descriptor_set(instance, data)

    def __delete__(
        self: Self,
        instance: LazyObject
    ) -> Never:
        raise TypeError("Cannot delete attributes of a lazy object")

    #def _can_override(
    #    self: Self,
    #    descriptor: LazyDescriptor
    #) -> bool:
    #    return (
    #        self._is_plural is descriptor._is_plural
    #        #and self._hasher is descriptor._hasher
    #        #and (self._freeze or not descriptor._freeze)
    #        and (
    #            self._element_type_annotation == descriptor._element_type_annotation
    #            or self._element_type is None
    #            or descriptor._element_type is None
    #            or issubclass(self._element_type, descriptor._element_type)
    #        )
    #    )
