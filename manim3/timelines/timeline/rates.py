from __future__ import annotations


from typing import (
    Never,
    Self
)

from ...constants.custom_typing import BoundaryT
from .rate import Rate


class Compose(Rate):
    __slots__ = ("_reversed_rates",)

    def __init__(
        self: Self,
        *rates: Rate
    ) -> None:
        super().__init__()
        self._reversed_rates: tuple[Rate, ...] = rates[::-1]

    def at(
        self: Self,
        t: float
    ) -> float:
        result = t
        for rate in self._reversed_rates:
            result = rate.at(result)
        return result

    def at_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> BoundaryT:
        result = boundary
        for rate in self._reversed_rates:
            result = rate.at_boundary(result)
        return result

    def is_increasing(
        self: Self
    ) -> bool:
        return all(
            rate.is_increasing()
            for rate in self._reversed_rates
        )


class LinearRate(Rate):
    __slots__ = ()

    def at(
        self: Self,
        t: float
    ) -> float:
        return t

    def at_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> BoundaryT:
        return boundary

    def is_increasing(
        self: Self
    ) -> bool:
        return True


class RushFromRate(Rate):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()

    def at(
        self: Self,
        t: float
    ) -> float:
        return (3.0 * t ** 5 - 10.0 * t ** 3 + 15.0 * t) / 8.0

    def at_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> BoundaryT:
        return boundary

    def is_increasing(
        self: Self
    ) -> bool:
        return True


class RushIntoRate(RushFromRate):
    __slots__ = ()

    def at(
        self: Self,
        t: float
    ) -> float:
        return super().at(t - 1.0) + 1.0


class SmoothRate(RushFromRate):
    __slots__ = ()

    def at(
        self: Self,
        t: float
    ) -> float:
        return (super().at(2.0 * t - 1.0) + 1.0) / 2.0


class RewindRate(Rate):
    __slots__ = ()

    def at(
        self: Self,
        t: float
    ) -> float:
        return 1.0 - t

    def at_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> BoundaryT:
        return 1 - boundary

    def is_increasing(
        self: Self
    ) -> bool:
        return False


class CycleRate(Rate):
    __slots__ = ()

    def at(
        self: Self,
        t: float
    ) -> float:
        remainder = t % 2.0
        return remainder if remainder < 1.0 else 2.0 - remainder

    def at_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> BoundaryT:
        return boundary

    def is_increasing(
        self: Self
    ) -> bool:
        return False


class RepeatRate(Rate):
    __slots__ = ()

    def at(
        self: Self,
        t: float
    ) -> float:
        return t % 1.0

    def at_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> BoundaryT:
        return boundary

    def is_increasing(
        self: Self
    ) -> bool:
        return False


class Rates:
    __slots__ = ()

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @classmethod
    def compose(
        cls: type[Self],
        *rates: Rate
    ) -> Compose:
        return Compose(*rates)

    @classmethod
    def linear(
        cls: type[Self]
    ) -> LinearRate:
        return LinearRate()

    @classmethod
    def rush_from(
        cls: type[Self]
    ) -> RushFromRate:
        return RushFromRate()

    @classmethod
    def rush_into(
        cls: type[Self]
    ) -> RushIntoRate:
        return RushIntoRate()

    @classmethod
    def smooth(
        cls: type[Self]
    ) -> SmoothRate:
        return SmoothRate()

    @classmethod
    def rewind(
        cls: type[Self]
    ) -> RewindRate:
        return RewindRate()

    @classmethod
    def cycle(
        cls: type[Self]
    ) -> CycleRate:
        return CycleRate()

    @classmethod
    def repeat(
        cls: type[Self]
    ) -> RepeatRate:
        return RepeatRate()
