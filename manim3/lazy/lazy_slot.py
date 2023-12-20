from __future__ import annotations


import weakref
from typing import (
    TYPE_CHECKING,
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
        "_elements",
        "_associated_slots"
    )

    def __init__(
        self: Self,
        descriptor: LazyDescriptor[T, DataT]
    ) -> None:
        super().__init__()
        self._descriptor_ref: weakref.ref[LazyDescriptor[T, DataT]] = weakref.ref(descriptor)
        self._elements: tuple[Memoized[T], ...] | None = None
        self._associated_slots: weakref.WeakSet[LazySlot] = weakref.WeakSet()

    def get_descriptor(
        self: Self
    ) -> LazyDescriptor[T, DataT]:
        assert (descriptor := self._descriptor_ref()) is not None
        return descriptor

    def get(
        self: Self
    ) -> tuple[Memoized[T], ...] | None:
        return self._elements

    def set(
        self: Self,
        elements: tuple[Memoized[T], ...],
        associated_slots: set[LazySlot]
    ) -> None:
        self._elements = elements
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
