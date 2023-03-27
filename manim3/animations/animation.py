__all__ = [
    "Animation",
    "RegroupItem",
    "RegroupVerb"
]


from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Iterable
)
import warnings

import numpy as np

from ..mobjects.mobject import Mobject
from ..utils.rate import RateUtils


class RegroupVerb(Enum):
    ADD = 1
    BECOMES = 0
    DISCARD = -1


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class RegroupItem:
    mobjects: Mobject | Iterable[Mobject]
    verb: RegroupVerb
    targets: Mobject | Iterable[Mobject]


class Animation(ABC):
    __slots__ = (
        "_time_animate_func",
        "_time_regroup_items",
        "_run_time"
    )

    def __init__(
        self,
        *,
        # Two arguments provided are `(alpha_0, alpha)`.
        alpha_animate_func: Callable[[float, float], None],
        alpha_regroup_items: list[tuple[float, RegroupItem]],
        #start_time: float,
        run_time: float | None,
        # (time / run_time) |-> alpha
        rate_func: Callable[[float], float] | None = None
    ) -> None:
        assert run_time is None or run_time >= 0.0
        if rate_func is None:
            rate_func = RateUtils.linear
        if run_time is not None:
            rate_func = RateUtils.compose(rate_func, lambda t: t / run_time)

        def time_animate_func(
            t0: float,
            t: float
        ) -> None:
            alpha_animate_func(rate_func(t0), rate_func(t))

        def alpha_to_time(
            alpha: float
        ) -> float:
            t = RateUtils.inverse(rate_func, alpha)
            if run_time is not None and t > run_time:
                if not np.isclose(t, run_time):
                    warnings.warn("`time_regroup_items` is not within `run_time`")
                t = run_time
            return t

        self._time_animate_func: Callable[[float, float], None] = time_animate_func
        self._time_regroup_items: list[tuple[float, RegroupItem]] = [
            (alpha_to_time(alpha), regroup_item)
            for alpha, regroup_item in alpha_regroup_items
        ]
        self._run_time: float | None = run_time
