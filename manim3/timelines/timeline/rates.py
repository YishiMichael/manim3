from __future__ import annotations


import functools
import operator
from abc import (
    ABC,
    abstractmethod
)
from typing import (
    Never,
    Self
)

import numpy as np
from scipy.interpolate import BSpline

from ...constants.constants import TAU
from ...constants.custom_typing import NP_xf8


class Rate(ABC):
    __slots__ = ("_is_increasing",)

    def __init__(
        self: Self,
        is_increasing: bool
    ) -> None:
        super().__init__()
        self._is_increasing: bool = is_increasing

    @abstractmethod
    def at(
        self: Self,
        t: float
    ) -> float:
        pass


class ComposeRate(Rate):
    __slots__ = ("_rates",)

    def __init__(
        self: Self,
        *rates: Rate
    ) -> None:
        super().__init__(is_increasing=all(rate._is_increasing for rate in rates))
        self._rates: tuple[Rate, ...] = rates

    def at(
        self: Self,
        time: float
    ) -> float:
        alpha = time
        for rate in reversed(self._rates):
            alpha = rate.at(alpha)
        return alpha


class ProductRate(Rate):
    __slots__ = ("_rates",)

    def __init__(
        self: Self,
        *rates: Rate
    ) -> None:
        super().__init__(is_increasing=all(rate._is_increasing for rate in rates))
        self._rates: tuple[Rate, ...] = rates

    def at(
        self: Self,
        time: float
    ) -> float:
        return functools.reduce(operator.mul, (
            rate.at(time)
            for rate in self._rates
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
        assert rate._is_increasing
        min_alpha = rate.at(min_time)
        max_alpha = rate.at(max_time)
        delta_time = max_time - min_time
        delta_alpha = max_alpha - min_alpha
        assert delta_time > 0.0
        assert delta_alpha > 0.0
        super().__init__(is_increasing=True)
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

    def __init__(
        self: Self
    ) -> None:
        super().__init__(is_increasing=True)

    def at(
        self: Self,
        time: float
    ) -> float:
        return time


class BezierRate(Rate):
    __slots__ = ("_curve",)

    def __init__(
        self: Self,
        values: NP_xf8
    ) -> None:
        super().__init__(is_increasing=bool(np.all(np.diff(values) >= 0.0)))
        self._curve: BSpline = BSpline(
            t=np.repeat(np.array((0.0, 1.0)), len(values)),
            c=values,
            k=len(values) - 1
        )

    def at(
        self: Self,
        time: float
    ) -> float:
        return float(self._curve(time))


class RewindRate(Rate):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        super().__init__(is_increasing=True)

    def at(
        self: Self,
        time: float
    ) -> float:
        return 1.0 - time


class ThereAndBackRate(Rate):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        super().__init__(is_increasing=False)

    def at(
        self: Self,
        time: float
    ) -> float:
        return 1.0 - abs(2.0 * time - 1.0)


class SinusoidalRate(Rate):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        super().__init__(is_increasing=False)

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
        super().__init__(is_increasing=False)
        self._periods: int = periods

    def at(
        self: Self,
        time: float
    ) -> float:
        return time * float(self._periods) % 1.0


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

    # Variant rates, borrowed from `3b1b/manim`.

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
        return cls.bezier(np.array((0.0, 0.0, pull_factor, pull_factor, 1.0, 1.0, 1.0)))

    @classmethod
    def overshoot(
        cls: type[Self],
        pull_factor: float = 1.5
    ) -> Rate:
        return cls.bezier(np.array((0.0, 0.0, pull_factor, pull_factor, 1.0, 1.0)))

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
