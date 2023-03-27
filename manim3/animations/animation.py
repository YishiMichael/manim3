__all__ = [
    "AlphaAnimation",
    "Animation",
    "RegroupItem",
    "RegroupVerb"
    #"SimpleAnimation"
]


from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Iterable
)

from ..mobjects.mobject import Mobject
from ..utils.rate import RateUtils


class RegroupVerb(Enum):
    ADD = 1
    DISCARD = -1


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class RegroupItem:
    mobjects: Mobject | None | Iterable[Mobject | None]  # `None` represents the scene.
    verb: RegroupVerb
    targets: Mobject | Iterable[Mobject]


class Animation(ABC):
    __slots__ = (
        "_animate_func",
        "_time_regroup_items",
        "_start_time",
        "_stop_time"
    )

    def __init__(
        self,
        *,
        # Two arguments provided are (t0, t).
        animate_func: Callable[[float, float], None],
        time_regroup_items: list[tuple[float, RegroupItem]],
        start_time: float,
        stop_time: float | None
    ) -> None:
        assert stop_time is None or stop_time >= start_time
        self._animate_func: Callable[[float, float], None] = animate_func
        self._time_regroup_items: list[tuple[float, RegroupItem]] = time_regroup_items
        self._start_time: float = start_time
        self._stop_time: float | None = stop_time


class AlphaAnimation(Animation):
    __slots__ = ()

    def __init__(
        self,
        *,
        # In terms of alpha instead of time.
        animate_func: Callable[[float, float], None],
        alpha_regroup_items: list[tuple[float, RegroupItem]],
        run_time: float,
        rate_func: Callable[[float], float] | None = None
    ) -> None:
        assert run_time > 0.0
        if rate_func is None:
            rate_func = RateUtils.smooth
        super().__init__(
            animate_func=lambda t0, t: animate_func(rate_func(t0 / run_time), rate_func(t / run_time)),
            time_regroup_items=[
                (rate_func(alpha) * run_time, regroup_item)
                for alpha, regroup_item in alpha_regroup_items
            ],
            start_time=0.0,
            stop_time=run_time
        )


#class SimpleAnimation(AlphaAnimation):
#    # Animation that does not contain any addition or removal of mobjects
#    def __init__(
#        self,
#        *,
#        animate_func: Callable[[float, float], None],
#        run_time: float,
#        rate_func: Callable[[float], float] | None = None
#    ):
#        super().__init__(
#            animate_func=animate_func,
#            mobject_addition_items=[],
#            mobject_removal_items=[],
#            run_time=run_time,
#            rate_func=rate_func
#        )
