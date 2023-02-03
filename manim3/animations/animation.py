#__all__ = ["Animation"]


from typing import Callable

from ..custom_typing import (
    Real,
    Vec3T
)
from ..mobjects.mobject import Mobject
#from ..mobjects.shape_mobject import ShapeMobject
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


class SimpleAnimation(AlphaAnimation):
    # Animation that does not contain any addition or removal of mobjects
    def __init__(
        self,
        *,
        animate_func: Callable[[Real, Real], None],
        run_time: Real,
        rate_func: Callable[[Real], Real] | None = None
    ):
        super().__init__(
            animate_func=animate_func,
            mobject_addition_items=[],
            mobject_removal_items=[],
            run_time=run_time,
            rate_func=rate_func
        )


class Shift(SimpleAnimation):
    # The interface is aligned with `Mobject.shift()`
    def __init__(
        self,
        mobject: Mobject,
        vector: Vec3T,
        *,
        coor_mask: Vec3T | None = None,
        broadcast: bool = True,
        run_time: Real = 1.0,
        rate_func: Callable[[Real], Real] | None = None
    ):
        def animate_func(alpha_0: Real, alpha: Real) -> None:
            mobject.shift(
                vector * (alpha - alpha_0),
                coor_mask=coor_mask,
                broadcast=broadcast
            )
        super().__init__(
            animate_func=animate_func,
            run_time=run_time,
            rate_func=rate_func
        )




#class Animation(Generic[MobjectType]):
#    #def __init__(
#    #    self,
#    #    #run_time: Real = 1.0
#    #):
#    #    #self.run_time: Real = 1.0
#    #    self.elapsed_time: Real = 0.0

#    def update_dt(self, mobject: MobjectType, dt: Real) -> None:
#        self.elapsed_time += dt
#        self.update(mobject, self.elapsed_time)

#    def start(self, initial_mobject: MobjectType) -> None:
#        self.elapsed_time: Real = 0.0  # TODO: outside __init__
#        #self.initial_mobject = mobject.copy()  # TODO: outside __init__

#    def update(self, mobject: MobjectType, t: Real) -> None:
#        pass

#    def expired(self) -> bool:
#        return False


#class DrawShape(Animation[ShapeMobject]):
#    def start(self, initial_mobject: ShapeMobject) -> None:
#        super().start(initial_mobject)
#        self.initial_shape = initial_mobject._shape_

#    def update(self, mobject: ShapeMobject, t: Real) -> None:
#        if t > 3.0:
#            return
#        #self.initial_mobject.path.partial_by_l_ratio(t / 3).skia_path.dump()
#        #self.initial_path.skia_path.dump()
#        #print(t)
#        #print(mobject._path_._l_final_)
#        mobject.set_shape(self.initial_shape.partial(0.0, t / 3.0))

#    def expired(self) -> bool:
#        return self.elapsed_time > 3.0
