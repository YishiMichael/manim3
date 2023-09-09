import weakref
from typing import (
    TYPE_CHECKING,
    Generic,
    Hashable,
    Iterator,
    TypeVar
)

if TYPE_CHECKING:
    from .lazy_descriptor import Registered


_T = TypeVar("_T")


class LazySlot(Generic[_T]):
    __slots__ = (
        "__weakref__",
        "_elements",
        "_parameter_key",
        "_linked_slots",
        "_is_writable"
    )

    def __init__(self) -> None:
        super().__init__()
        self._elements: "tuple[Registered[_T], ...] | None" = None
        self._parameter_key: "Registered[Hashable] | None" = None
        self._linked_slots: weakref.WeakSet[LazySlot] = weakref.WeakSet()
        self._is_writable: bool = True

    def _get(self) -> "tuple[Registered[_T], ...] | None":
        return self._elements

    def _set(
        self,
        elements: "tuple[Registered[_T], ...]",
        parameter_key: "Registered[Hashable] | None",
        linked_slots: "set[LazySlot]"
    ) -> None:
        self._elements = elements
        self._parameter_key = parameter_key
        assert not self._linked_slots
        self._linked_slots.update(linked_slots)
        for slot in linked_slots:
            slot._linked_slots.add(self)

    def _expire(self) -> None:
        self._elements = None
        for slot in self._linked_slots:
            slot._linked_slots.remove(self)
        self._linked_slots.clear()

    def _iter_linked_slots(self) -> "Iterator[LazySlot]":
        return iter(set(self._linked_slots))
