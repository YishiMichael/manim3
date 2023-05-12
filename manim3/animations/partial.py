from typing import (
    Any,
    Callable,
    ClassVar
)

from ..animations.animation import Animation
from ..lazy.lazy import LazyVariableDescriptor
from ..mobjects.mobject import Mobject
from ..mobjects.shape_mobject import ShapeMobject
from ..mobjects.stroke_mobject import StrokeMobject
from ..shape.line_string import MultiLineString
from ..shape.shape import Shape
from ..utils.rate import RateUtils


class PartialAnimation(Animation):
    __slots__ = ()

    _partial_methods: ClassVar[dict[LazyVariableDescriptor, Callable[[Any, float, float], Any]]] = {
        ShapeMobject._shape_: Shape.partial,
        StrokeMobject._multi_line_string_: MultiLineString.partial
    }

    def __init__(
        self,
        mobject: Mobject,
        alpha_to_boundary_values: Callable[[float], tuple[float, float]],
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            run_time=run_time,
            relative_rate=RateUtils.adjust(rate_func, run_time_scale=run_time),
            updater=updater
        )
