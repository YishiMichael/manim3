#__all__ = [
#    "Shift",
#    "Scale"
#]


#from typing import Callable

#import numpy as np

#from ..animations.animation import SimpleAnimation
#from ..custom_typing import (
#    Real,
#    Vec3T
#)
#from ..mobjects.mobject import Mobject
#from ..utils.space import SpaceUtils


#class Shift(SimpleAnimation):
#    # The interface is aligned with `Mobject.shift()`
#    def __init__(
#        self,
#        mobject: Mobject,
#        vector: Vec3T,
#        *,
#        coor_mask: Vec3T | None = None,
#        broadcast: bool = True,
#        run_time: Real = 1.0,
#        rate_func: Callable[[Real], Real] | None = None
#    ):
#        def animate_func(alpha_0: Real, alpha: Real) -> None:
#            mobject.shift(
#                vector * (alpha - alpha_0),
#                coor_mask=coor_mask,
#                broadcast=broadcast
#            )
#        super().__init__(
#            animate_func=animate_func,
#            run_time=run_time,
#            rate_func=rate_func
#        )


#class Scale(SimpleAnimation):
#    # The interface is aligned with `Mobject.scale()`
#    def __init__(
#        self,
#        mobject: Mobject,
#        factor: Real | Vec3T,
#        *,
#        about_point: Vec3T | None = None,
#        about_edge: Vec3T | None = None,
#        broadcast: bool = True,
#        run_time: Real = 1.0,
#        rate_func: Callable[[Real], Real] | None = None
#    ):
#        about_point = mobject._calculate_about_point(
#            about_point=about_point,
#            about_edge=about_edge,
#            broadcast=broadcast
#        )
#        factor_0: float | Vec3T = 1.0 if isinstance(factor, Real) else np.ones(3)
#        def animate_func(alpha_0: Real, alpha: Real) -> None:
#            mobject.scale(
#                SpaceUtils.lerp(factor_0, factor, alpha) / SpaceUtils.lerp(factor_0, factor, alpha_0),
#                about_point=about_point,
#                broadcast=broadcast
#            )
#        super().__init__(
#            animate_func=animate_func,
#            run_time=run_time,
#            rate_func=rate_func
#        )
