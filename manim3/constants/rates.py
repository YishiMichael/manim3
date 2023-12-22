from __future__ import annotations
import itertools


import math
from functools import reduce
from typing import (
    Never,
    Self
)

from .constants import TAU
from .custom_typing import RateType


class Rates:
    __slots__ = ()

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @classmethod
    def compose(
        cls: type[Self],
        *rates: RateType
    ) -> RateType:
        return lambda time: reduce(lambda t, rate: rate(t), reversed(rates), time)

    @classmethod
    def sum(
        cls: type[Self],
        *rates: RateType
    ) -> RateType:
        return lambda time: reduce(float.__add__, (rate(time) for rate in rates), 0.0)

    @classmethod
    def product(
        cls: type[Self],
        *rates: RateType
    ) -> RateType:
        return lambda time: reduce(float.__mul__, (rate(time) for rate in rates), 1.0)

    @classmethod
    def clip(
        cls: type[Self],
        rate: RateType,
        delta_time: float = 1.0,
        delta_alpha: float = 1.0,
        start_time: float = 0.0,
        start_alpha: float = 0.0
    ) -> RateType:
        return lambda time: (rate(time * delta_time + start_time) - start_alpha) / delta_alpha

    @classmethod
    def linear(
        cls: type[Self]
    ) -> RateType:
        return lambda time: time

    @classmethod
    def bezier(
        cls: type[Self],
        *anchors: float
    ) -> RateType:
        degree = len(anchors) - 1
        return lambda time: sum(
            math.comb(degree, k) * pow(1.0 - time, degree - k) * pow(time, k) * anchor
            for k, anchor in enumerate(anchors)
        )

    @classmethod
    def rewind(
        cls: type[Self]
    ) -> RateType:
        return lambda time: 1.0 - time

    @classmethod
    def there_and_back(
        cls: type[Self]
    ) -> RateType:
        return lambda time: 1.0 - abs(2.0 * time - 1.0)

    @classmethod
    def sinusoidal(
        cls: type[Self]
    ) -> RateType:
        return lambda time: math.sin(TAU * time)

    @classmethod
    def repeat(
        cls: type[Self],
        periods: int
    ) -> RateType:
        return lambda time: periods * time % 1.0

    # Variant rates, borrowed from `3b1b/manim`.

    @classmethod
    def smoothstep(
        cls: type[Self],
        degree: int = 1
    ) -> RateType:
        return cls.bezier(*itertools.repeat(0.0, degree + 1), *itertools.repeat(1.0, degree + 1))

    @classmethod
    def smooth(
        cls: type[Self]
    ) -> RateType:
        return cls.smoothstep(2)

    @classmethod
    def rush_from(
        cls: type[Self]
    ) -> RateType:
        return cls.clip(cls.smooth(), delta_time=0.5, delta_alpha=0.5, start_time=0.5, start_alpha=0.5)

    @classmethod
    def rush_into(
        cls: type[Self]
    ) -> RateType:
        return cls.clip(cls.smooth(), delta_time=0.5, delta_alpha=0.5)

    @classmethod
    def overshoot(
        cls: type[Self]
    ) -> RateType:
        return cls.bezier(0.0, 0.0, 1.5, 1.5, 1.0, 1.0)

    @classmethod
    def undershoot(
        cls: type[Self]
    ) -> RateType:
        return cls.bezier(0.0, 0.0, -0.5, -0.5, 1.0, 1.0)

    @classmethod
    def there_and_back_smoothly(
        cls: type[Self]
    ) -> RateType:
        return cls.compose(cls.smooth(), cls.there_and_back())

    @classmethod
    def wiggle(
        cls: type[Self]
    ) -> RateType:
        return cls.product(cls.there_and_back_smoothly(), cls.sinusoidal())
