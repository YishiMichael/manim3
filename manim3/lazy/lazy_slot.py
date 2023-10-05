from __future__ import annotations


import weakref
from typing import (
    TYPE_CHECKING,
    Hashable,
    Iterator,
    Self
)

if TYPE_CHECKING:
    from .lazy_descriptor import Registered


class LazySlot[T]:
    __slots__ = (
        "__weakref__",
        "_elements",
        "_parameter_key",
        "_linked_slots",
        "_is_writable"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._elements: tuple[Registered[T], ...] | None = None
        self._parameter_key: Registered[Hashable] | None = None
        self._linked_slots: weakref.WeakSet[LazySlot] = weakref.WeakSet()
        self._is_writable: bool = True

    def _get(
        self: Self
    ) -> tuple[Registered[T], ...] | None:
        return self._elements

    def _set(
        self: Self,
        elements: tuple[Registered[T], ...],
        parameter_key: Registered[Hashable] | None,
        linked_slots: set[LazySlot]
    ) -> None:
        self._elements = elements
        self._parameter_key = parameter_key
        assert not self._linked_slots
        self._linked_slots.update(linked_slots)
        for slot in linked_slots:
            slot._linked_slots.add(self)

    def _expire(
        self: Self
    ) -> None:
        self._elements = None
        for slot in self._linked_slots:
            slot._linked_slots.remove(self)
        self._linked_slots.clear()

    def _iter_linked_slots(
        self: Self
    ) -> Iterator[LazySlot]:
        return iter(set(self._linked_slots))
