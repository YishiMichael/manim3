__all__ = [
    "AlphaAnimation",
    "Animation"
    #"SimpleAnimation"
]


from abc import ABC
from typing import Callable

from ..mobjects.mobject import Mobject
from ..utils.rate import RateUtils


class Animation(ABC):
    __slots__ = (
        "_animate_func",
        "_mobject_addition_items",
        "_mobject_removal_items",
        "_start_time",
        "_stop_time"
    )

    def __init__(
        self,
        *,
        # Two arguments provided are (t0, t).
        animate_func: Callable[[float, float], None],
        # `tuple[float, Mobject, Mobject | None]` stands for a (time, mobject, parent) triplet. `None` represents the scene.
        mobject_addition_items: list[tuple[float, Mobject, Mobject | None]],
        mobject_removal_items: list[tuple[float, Mobject, Mobject | None]],
        start_time: float,
        stop_time: float | None
    ) -> None:
        assert stop_time is None or stop_time >= start_time
        self._animate_func: Callable[[float, float], None] = animate_func
        self._mobject_addition_items: list[tuple[float, Mobject, Mobject | None]] = mobject_addition_items
        self._mobject_removal_items: list[tuple[float, Mobject, Mobject | None]] = mobject_removal_items
        self._start_time: float = start_time
        self._stop_time: float | None = stop_time


class AlphaAnimation(Animation):
    __slots__ = ()

    def __init__(
        self,
        *,
        # In terms of alpha instead of time.
        animate_func: Callable[[float, float], None],
        mobject_addition_items: list[tuple[float, Mobject, Mobject | None]],
        mobject_removal_items: list[tuple[float, Mobject, Mobject | None]],
        run_time: float,
        rate_func: Callable[[float], float] | None = None
    ) -> None:
        assert run_time > 0.0
        if rate_func is None:
            rate_func = RateUtils.smooth
        super().__init__(
            animate_func=lambda t0, t: animate_func(rate_func(t0 / run_time), rate_func(t / run_time)),
            mobject_addition_items=[
                (rate_func(alpha) * run_time, mobject, parent)
                for alpha, mobject, parent in mobject_addition_items
            ],
            mobject_removal_items=[
                (rate_func(alpha) * run_time, mobject, parent)
                for alpha, mobject, parent in mobject_removal_items
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
