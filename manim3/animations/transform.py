__all__ = ["ShapeMobjectTransform"]


from typing import (
    Any,
    Callable
)

from ..animations.animation import AlphaAnimation
from ..custom_typing import Real
from ..mobjects.shape_mobject import ShapeMobject
from ..utils.lazy import (
    LazyData,
    lazy_basedata
)
from ..utils.space import SpaceUtils
from ..utils.shape import Shape


class ShapeMobjectTransform(AlphaAnimation):
    def __init__(
        self,
        start_mobject: ShapeMobject,
        stop_mobject: ShapeMobject,
        *,
        run_time: Real = 3.0,
        rate_func: Callable[[Real], Real] | None = None
    ):
        intermediate_mobject = start_mobject.copy()
        def animate_func(alpha_0: Real, alpha: Real) -> None:
            for basedata_descr, interpolate_method in self._get_interpolate_methods().items():
                if (start_basedata := basedata_descr._get_data(start_mobject)) \
                        is (stop_basedata := basedata_descr._get_data(stop_mobject)):
                    continue
                basedata_descr.__set__(intermediate_mobject, LazyData(interpolate_method(
                    start_basedata.data, stop_basedata.data, alpha
                )))

        super().__init__(
            animate_func=animate_func,
            mobject_addition_items=[
                (0.0, intermediate_mobject, None),
                (1.0, stop_mobject, None)
            ],  # TODO: parents?
            mobject_removal_items=[
                (0.0, start_mobject, None),
                (1.0, intermediate_mobject, None)
            ],
            run_time=run_time,
            rate_func=rate_func
        )

    @classmethod
    def _get_interpolate_methods(cls) -> dict[lazy_basedata[ShapeMobject, Any], Callable[[Any, Any, Real], Any]]:
        return {
            ShapeMobject._shape_: Shape.interpolate_method,
            ShapeMobject._model_matrix_: SpaceUtils.rotational_interpolate,
            ShapeMobject._color_: SpaceUtils.lerp,
            ShapeMobject._opacity_: SpaceUtils.lerp,
            ShapeMobject._ambient_strength_: SpaceUtils.lerp,
            ShapeMobject._specular_strength_: SpaceUtils.lerp,
            ShapeMobject._shininess_: SpaceUtils.lerp
        }
