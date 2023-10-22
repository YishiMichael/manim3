from __future__ import annotations


import weakref
from typing import (
    TYPE_CHECKING,
    Callable,
    Hashable,
    Self
)

from .registration import (
    Registered,
    Registration
)

if TYPE_CHECKING:
    from .lazy_object import LazyObject
    from .lazy_routine import LazyRoutine
    from .lazy_slot import LazySlot


class LazyOverriding[T, DataT]:
    __slots__ = (
        "_name",
        "_is_plural",
        "_hasher",
        "_freezer",
        "_element_registration",
        "_routines"
        #"_element_type_hints"
    )

    def __init__(
        self: Self,
        name: str,
        is_plural: bool,
        hasher: Callable[[T], Hashable],
        freezer: Callable[[T], None]
    ) -> None:
        super().__init__()
        self._name: str = name
        self._is_plural: bool = is_plural
        self._hasher: Callable[[T], Hashable] = hasher
        self._freezer: Callable[[T], None] = freezer
        self._element_registration: Registration[Hashable, T] = Registration()
        self._routines: weakref.WeakValueDictionary[type[LazyObject], LazyRoutine[T, DataT]] = weakref.WeakValueDictionary()
        #self._element_type_hints: dict[type[LazyObject], TypeHint] = {}
        #self._element_annotations: dict[type[LazyObject], Any] = {}
        #self._element_lazy_cls: dict[type[LazyObject], type[LazyObject] | None] = {}
#
    def _register_elements(
        self: Self,
        elements: tuple[T, ...],
        freeze: bool
    ) -> tuple[Registered[T], ...]:
        if freeze:
            freezer = self._freezer
            for element in elements:
                freezer(element)
        element_registration = self._element_registration
        hasher = self._hasher
        return tuple(
            element_registration.register(hasher(element), element)
            for element in elements
        )

    def get_slot(
        self: Self,
        instance: LazyObject
    ) -> LazySlot[T]:
        return instance._get_slot(self._name)

    #def set_slot(
    #    self: Self,
    #    instance: LazyObject,
    #    slot: LazySlot[T]
    #) -> None:
    #    instance._lazy_slots[self._name] = slot
