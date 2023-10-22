from __future__ import annotations


import weakref
from typing import (
    Hashable,
    Self
)


class Registered[T]:
    __slots__ = (
        "__weakref__",
        "_value"
    )

    def __init__(
        self: Self,
        value: T
    ) -> None:
        super().__init__()
        self._value: T = value


class Registration[KT: Hashable, VT](weakref.WeakValueDictionary[KT, Registered[VT]]):
    __slots__ = ()

    def register(
        self: Self,
        key: KT,
        value: VT
    ) -> Registered[VT]:
        if (registered_value := self.get(key)) is None:
            registered_value = Registered(value)
            self[key] = registered_value
        return registered_value
