__all__ = [
    "Animation",
    "AlphaAnimation",
    #"SimpleAnimation"
]


from typing import Callable

from ..custom_typing import Real
from ..mobjects.mobject import Mobject
from ..utils.rate import RateUtils


class Animation:
    def __init__(
        self,
        *,
        # Two arguments provided are (t0, t).
        animate_func: Callable[[Real, Real], None],
        # `tuple[Real, Mobject, Mobject | None]` stands for a (time, mobject, parent) triplet. `None` represents the scene.
        mobject_addition_items: list[tuple[Real, Mobject, Mobject | None]],
        mobject_removal_items: list[tuple[Real, Mobject, Mobject | None]],
        start_time: Real,
        stop_time: Real | None
    ):
        assert stop_time is None or stop_time >= start_time
        self._animate_func: Callable[[Real, Real], None] = animate_func
        self._mobject_addition_items: list[tuple[Real, Mobject, Mobject | None]] = mobject_addition_items
        self._mobject_removal_items: list[tuple[Real, Mobject, Mobject | None]] = mobject_removal_items
        self._start_time: Real = start_time
        self._stop_time: Real | None = stop_time


class AlphaAnimation(Animation):
    def __init__(
        self,
        *,
        # In terms of alpha instead of time
        animate_func: Callable[[Real, Real], None],
        mobject_addition_items: list[tuple[Real, Mobject, Mobject | None]],
        mobject_removal_items: list[tuple[Real, Mobject, Mobject | None]],
        run_time: Real,
        rate_func: Callable[[Real], Real] | None = None
    ):
        assert run_time > 0.0
        if rate_func is None:
            rate_func = RateUtils.smooth
        super().__init__(
            animate_func=lambda t0, t: animate_func(rate_func(t0 / run_time), rate_func(t / run_time)),
            mobject_addition_items=[(rate_func(t / run_time), mobject, parent) for t, mobject, parent in mobject_addition_items],
            mobject_removal_items=[(rate_func(t / run_time), mobject, parent) for t, mobject, parent in mobject_removal_items],
            start_time=0.0,
            stop_time=run_time
        )


#class SimpleAnimation(AlphaAnimation):
#    # Animation that does not contain any addition or removal of mobjects
#    def __init__(
#        self,
#        *,
#        animate_func: Callable[[Real, Real], None],
#        run_time: Real,
#        rate_func: Callable[[Real], Real] | None = None
#    ):
#        super().__init__(
#            animate_func=animate_func,
#            mobject_addition_items=[],
#            mobject_removal_items=[],
#            run_time=run_time,
#            rate_func=rate_func
#        )
