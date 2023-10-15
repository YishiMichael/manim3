from __future__ import annotations


import functools
import operator
from typing import (
    Never,
    Self
)

import numpy as np
from scipy.interpolate import BSpline

from ...constants.constants import TAU
from ...constants.custom_typing import (
    BoundaryT,
    NP_xf8
)
from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
from .rate import Rate


class ComposeRate(Rate):
    __slots__ = ()

    def __init__(
        self: Self,
        *rates: Rate
    ) -> None:
        super().__init__()
        self._reversed_rates_ = tuple(reversed(rates))

    @Lazy.variable_collection()
    @staticmethod
    def _reversed_rates_() -> tuple[Rate, ...]:
        return ()

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _is_increasing_(
        reversed_rates__is_increasing: tuple[bool, ...]
    ) -> bool:
        return all(reversed_rates__is_increasing)

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _boundaries_(
        reversed_rates__boundaries: tuple[tuple[BoundaryT, BoundaryT], ...]
    ) -> tuple[BoundaryT, BoundaryT]:
        boundaries = (0, 1)
        for rate_boundary_0, rate_boundary_1 in reversed_rates__boundaries:
            boundaries = (boundaries[rate_boundary_0], boundaries[rate_boundary_1])
        return boundaries

    def at(
        self: Self,
        time: float
    ) -> float:
        alpha = time
        for rate in self._reversed_rates_:
            alpha = rate.at(alpha)
        return alpha

    #def at_boundary(
    #    self: Self,
    #    boundary: BoundaryT
    #) -> BoundaryT:
    #    result = boundary
    #    for rate in self._reversed_rates:
    #        result = rate.at_boundary(result)
    #    return result

    #def is_increasing(
    #    self: Self
    #) -> bool:
    #    return all(
    #        rate.is_increasing()
    #        for rate in self._reversed_rates
    #    )


class ProductRate(Rate):
    __slots__ = ()

    def __init__(
        self: Self,
        *rates: Rate
    ) -> None:
        super().__init__()
        self._rates_ = rates

    @Lazy.variable_collection()
    @staticmethod
    def _rates_() -> tuple[Rate, ...]:
        return ()

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _is_increasing_(
        rates__is_increasing: tuple[bool, ...]
    ) -> bool:
        return all(rates__is_increasing)

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _boundaries_(
        rates__boundaries: tuple[tuple[BoundaryT, BoundaryT], ...]
    ) -> tuple[BoundaryT, BoundaryT]:
        boundary_0, boundary_1 = (0, 1)
        for rate_boundary_0, rate_boundary_1 in rates__boundaries:
            boundary_0, boundary_1 = (boundary_0 and rate_boundary_0, boundary_1 and rate_boundary_1)
        return (boundary_0, boundary_1)

    def at(
        self: Self,
        time: float
    ) -> float:
        return functools.reduce(operator.mul, (
            rate.at(time)
            for rate in self._rates_
        ), 1.0)


class ClipRate(Rate):
    __slots__ = (
        "_rate",
        "_min_time",
        "_delta_time",
        "_min_alpha",
        "_delta_alpha"
    )

    def __init__(
        self: Self,
        rate: Rate,
        min_time: float,
        max_time: float
    ) -> None:
        assert rate._is_increasing_
        min_alpha = rate.at(min_time)
        max_alpha = rate.at(max_time)
        delta_time = max_time - min_time
        delta_alpha = max_alpha - min_alpha
        assert delta_time > 0.0
        assert delta_alpha > 0.0
        super().__init__()
        self._rate: Rate = rate
        self._min_time: float = min_time
        self._delta_time: float = delta_time
        self._min_alpha: float = min_alpha
        self._delta_alpha: float = delta_alpha

    def at(
        self: Self,
        time: float
    ) -> float:
        return (self._rate.at(time * self._delta_time + self._min_time) - self._min_alpha) / self._delta_alpha


class LinearRate(Rate):
    __slots__ = ()

    def at(
        self: Self,
        time: float
    ) -> float:
        return time

    #def at_boundary(
    #    self: Self,
    #    boundary: BoundaryT
    #) -> BoundaryT:
    #    return boundary

    #def is_increasing(
    #    self: Self
    #) -> bool:
    #    return True


class BezierRate(Rate):
    __slots__ = ()

    def __init__(
        self,
        values: NP_xf8
    ) -> None:
        super().__init__()
        self._values_ = values

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _values_() -> NP_xf8:
        return np.zeros((0,))

    @Lazy.property()
    @staticmethod
    def _curve_(
        values: NP_xf8
    ) -> BSpline:
        return SpaceUtils.bezier(values)

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _is_increasing_(
        values: NP_xf8
    ) -> bool:
        return bool(np.all(np.diff(values) >= 0.0))

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _boundaries_(
        values: NP_xf8
    ) -> tuple[BoundaryT, BoundaryT]:
        boundary_0 = int(np.rint(values[0]))
        boundary_1 = int(np.rint(values[-1]))
        assert boundary_0 in (0, 1) and boundary_1 in (0, 1)
        return (boundary_0, boundary_1)

    def at(
        self: Self,
        time: float
    ) -> float:
        return float(self._curve_(np.array(time)))


#class RushFromRate(Rate):
#    __slots__ = ()

#    def __init__(self) -> None:
#        super().__init__()

#    def at(
#        self: Self,
#        t: float
#    ) -> float:
#        return (3.0 * t ** 5 - 10.0 * t ** 3 + 15.0 * t) / 8.0

#    def at_boundary(
#        self: Self,
#        boundary: BoundaryT
#    ) -> BoundaryT:
#        return boundary

#    def is_increasing(
#        self: Self
#    ) -> bool:
#        return True


#class RushIntoRate(RushFromRate):
#    __slots__ = ()

#    def at(
#        self: Self,
#        t: float
#    ) -> float:
#        return super().at(t - 1.0) + 1.0


#class SmoothRate(RushFromRate):
#    __slots__ = ()

#    def at(
#        self: Self,
#        t: float
#    ) -> float:
#        return (super().at(2.0 * t - 1.0) + 1.0) / 2.0


class RewindRate(Rate):
    __slots__ = ()

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _is_increasing_() -> bool:
        return False

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _boundaries_() -> tuple[BoundaryT, BoundaryT]:
        return (1, 0)

    def at(
        self: Self,
        time: float
    ) -> float:
        return 1.0 - time

    #def at_boundary(
    #    self: Self,
    #    boundary: BoundaryT
    #) -> BoundaryT:
    #    return 1 - boundary

    #def is_increasing(
    #    self: Self
    #) -> bool:
    #    return False


class ThereAndBackRate(Rate):
    __slots__ = ()

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _is_increasing_() -> bool:
        return False

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _boundaries_() -> tuple[BoundaryT, BoundaryT]:
        return (0, 0)

    def at(
        self: Self,
        time: float
    ) -> float:
        return 1.0 - abs(2.0 * time - 1.0)

    #def at_boundary(
    #    self: Self,
    #    boundary: BoundaryT
    #) -> BoundaryT:
    #    return boundary

    #def is_increasing(
    #    self: Self
    #) -> bool:
    #    return False


class SinusoidalRate(Rate):
    __slots__ = ()

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _is_increasing_() -> bool:
        return False

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _boundaries_() -> tuple[BoundaryT, BoundaryT]:
        return (0, 0)

    def at(
        self: Self,
        time: float
    ) -> float:
        return float(np.sin(time * TAU))


class RepeatRate(Rate):
    __slots__ = ("_periods",)

    def __init__(
        self: Self,
        periods: int
    ) -> None:
        assert periods > 0
        super().__init__()
        self._periods: int = periods

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _is_increasing_() -> bool:
        return False

    def at(
        self: Self,
        time: float
    ) -> float:
        return time * float(self._periods) % 1.0

    #def at_boundary(
    #    self: Self,
    #    boundary: BoundaryT
    #) -> BoundaryT:
    #    return boundary

    #def is_increasing(
    #    self: Self
    #) -> bool:
    #    return False


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
    ) -> ComposeRate:
        return ComposeRate(*rates)

    @classmethod
    def product(
        cls: type[Self],
        *rates: Rate
    ) -> ProductRate:
        return ProductRate(*rates)

    @classmethod
    def clip(
        cls: type[Self],
        rate: Rate,
        min_time: float,
        max_time: float
    ) -> ClipRate:
        return ClipRate(rate, min_time, max_time)

    @classmethod
    def linear(
        cls: type[Self]
    ) -> LinearRate:
        return LinearRate()

    @classmethod
    def bezier(
        cls: type[Self],
        values: NP_xf8
    ) -> BezierRate:
        return BezierRate(values)

    @classmethod
    def rewind(
        cls: type[Self]
    ) -> RewindRate:
        return RewindRate()

    @classmethod
    def there_and_back(
        cls: type[Self]
    ) -> ThereAndBackRate:
        return ThereAndBackRate()

    @classmethod
    def sinusoidal(
        cls: type[Self]
    ) -> SinusoidalRate:
        return SinusoidalRate()

    @classmethod
    def repeat(
        cls: type[Self],
        periods: int = 1
    ) -> RepeatRate:
        return RepeatRate(periods)

    # Variant rates, borrowed from 3b1b/manim.

    @classmethod
    def smooth(
        cls: type[Self]
    ) -> Rate:
        return cls.bezier(np.array((0.0, 0.0, 0.0, 1.0, 1.0, 1.0)))

    @classmethod
    def rush_from(
        cls: type[Self]
    ) -> Rate:
        return cls.clip(cls.smooth(), 0.5, 1.0)

    @classmethod
    def rush_into(
        cls: type[Self]
    ) -> Rate:
        return cls.clip(cls.smooth(), 0.0, 0.5)

    @classmethod
    def running_start(
        cls: type[Self],
        pull_factor: float = -0.5
    ) -> Rate:
        return cls.bezier(np.array((0, 0, pull_factor, pull_factor, 1, 1, 1)))

    @classmethod
    def overshoot(
        cls: type[Self],
        pull_factor: float = 1.5
    ) -> Rate:
        return cls.bezier(np.array((0, 0, pull_factor, pull_factor, 1, 1)))

    @classmethod
    def there_and_back_smoothly(
        cls: type[Self]
    ) -> Rate:
        return cls.compose(cls.smooth(), cls.there_and_back())

    @classmethod
    def wiggle(
        cls: type[Self]
    ) -> Rate:
        return cls.product(cls.there_and_back_smoothly(), cls.sinusoidal())
