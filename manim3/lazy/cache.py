from __future__ import annotations


import weakref
from typing import (
    Hashable,
    Self
)


class Cache[KT: Hashable, VT](weakref.WeakKeyDictionary[KT, VT]):
    __slots__ = ("_capacity",)

    def __init__(
        self: Self,
        capacity: int
    ) -> None:
        super().__init__()
        self._capacity: int = capacity

    def set(
        self: Self,
        key: KT,
        value: VT
    ) -> None:
        assert key not in self
        if len(self) == self._capacity:
            self.pop(next(iter(self)))
        self[key] = value
