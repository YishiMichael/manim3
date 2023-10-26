from __future__ import annotations


import weakref
from typing import (
    TYPE_CHECKING,
    Hashable,
    Iterator,
    Self
)

if TYPE_CHECKING:
    from .lazy_descriptor import (
        LazyDescriptor,
        Memoized
    )


class LazySlot[T, DataT]:
    __slots__ = (
        "__weakref__",
        "_descriptor_ref",
        "_is_writable",
        "_elements",
        "_parameter_key",
        "_associated_slots"
    )

    def __init__(
        self: Self,
        descriptor: LazyDescriptor[T, DataT]
    ) -> None:
        super().__init__()
        self._descriptor_ref: weakref.ref[LazyDescriptor[T, DataT]] = weakref.ref(descriptor)
        self._is_writable: bool = not descriptor._is_property
        self._elements: tuple[Memoized[T], ...] | None = None
        self._parameter_key: Memoized[Hashable] | None = None
        self._associated_slots: weakref.WeakSet[LazySlot] = weakref.WeakSet()

    def get_descriptor(
        self: Self
    ) -> LazyDescriptor[T, DataT]:
        assert (descriptor := self._descriptor_ref()) is not None
        return descriptor

    def disable_writability(
        self: Self
    ) -> None:
        self._is_writable = False

    def check_writability(
        self: Self
    ) -> None:
        assert self._is_writable, "Slot is not writable"

    def get(
        self: Self
    ) -> tuple[Memoized[T], ...] | None:
        return self._elements

    def set(
        self: Self,
        elements: tuple[Memoized[T], ...],
        parameter_key: Memoized[Hashable] | None,
        associated_slots: set[LazySlot]
    ) -> None:
        self._elements = elements
        self._parameter_key = parameter_key
        assert not self._associated_slots
        self._associated_slots.update(associated_slots)
        for slot in associated_slots:
            slot._associated_slots.add(self)

    def expire(
        self: Self
    ) -> None:
        self._elements = None
        for slot in self._associated_slots:
            slot._associated_slots.remove(self)
        self._associated_slots.clear()

    def iter_associated_slots(
        self: Self
    ) -> Iterator[LazySlot]:
        return iter(set(self._associated_slots))
