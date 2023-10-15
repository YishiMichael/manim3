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
        "_associated_slots",
        "_is_writable"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._elements: tuple[Registered[T], ...] | None = None
        self._parameter_key: Registered[Hashable] | None = None
        self._associated_slots: weakref.WeakSet[LazySlot] = weakref.WeakSet()
        self._is_writable: bool = True

    def get(
        self: Self
    ) -> tuple[Registered[T], ...] | None:
        return self._elements

    def set(
        self: Self,
        elements: tuple[Registered[T], ...],
        parameter_key: Registered[Hashable] | None,
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
