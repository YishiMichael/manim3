from __future__ import annotations


from abc import (
    ABC,
    abstractmethod
)
from typing import Self


class Condition(ABC):
    __slots__ = ()

    @abstractmethod
    def judge(
        self: Self
    ) -> bool:
        pass
