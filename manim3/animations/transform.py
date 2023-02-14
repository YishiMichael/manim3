__all__ = ["Transform"]


import itertools as it
from typing import (
    Any,
    Callable,
    ClassVar
)

from ..animations.animation import AlphaAnimation
from ..custom_typing import Real
from ..mobjects.shape_mobject import ShapeMobject
from ..mobjects.stroke_mobject import StrokeMobject
from ..utils.lazy import (
    NewData,
    LazyBasedata
)
from ..utils.space import SpaceUtils
from ..utils.shape import (
    MultiLineString3D,
    Shape
)


class Transform(AlphaAnimation):
    @staticmethod
    def shape_interpolate_method(
        shape_0: Shape,
        shape_1: Shape,
        alpha: Real
    ) -> Shape:
        return shape_0.interpolate_shape(shape_1, alpha, has_mending=True)

    @staticmethod
    def stroke_interpolate_method(
        multi_line_string_0: MultiLineString3D,
        multi_line_string_1: MultiLineString3D,
        alpha: Real
    ) -> MultiLineString3D:
        return multi_line_string_0.interpolate_shape(multi_line_string_1, alpha, has_mending=False)

    _SHAPE_INTERPOLATE_METHODS: ClassVar[dict[LazyBasedata[ShapeMobject, Any], Callable[[Any, Any, Real], Any]]] = {
        ShapeMobject._shape_: shape_interpolate_method,
        ShapeMobject._model_matrix_: SpaceUtils.rotational_interpolate,
        ShapeMobject._color_: SpaceUtils.lerp,
        ShapeMobject._opacity_: SpaceUtils.lerp,
        ShapeMobject._ambient_strength_: SpaceUtils.lerp,
        ShapeMobject._specular_strength_: SpaceUtils.lerp,
        ShapeMobject._shininess_: SpaceUtils.lerp
    }

    _STROKE_INTERPOLATE_METHODS: ClassVar[dict[LazyBasedata[StrokeMobject, Any], Callable[[Any, Any, Real], Any]]] = {
        StrokeMobject._multi_line_string_3d_: stroke_interpolate_method,
        StrokeMobject._model_matrix_: SpaceUtils.rotational_interpolate,
        StrokeMobject._color_: SpaceUtils.lerp,
        StrokeMobject._opacity_: SpaceUtils.lerp,
        StrokeMobject._width_: SpaceUtils.lerp,
        StrokeMobject._color_: SpaceUtils.lerp,
        StrokeMobject._opacity_: SpaceUtils.lerp,
        StrokeMobject._dilate_: SpaceUtils.lerp
    }

    def __init__(
        self,
        start_mobject: ShapeMobject,
        stop_mobject: ShapeMobject,
        *,
        run_time: Real = 2.0,
        rate_func: Callable[[Real], Real] | None = None
    ):
        intermediate_mobject = stop_mobject.copy()

        start_stroke_mobjects: list[StrokeMobject] = start_mobject._stroke_mobjects[:]
        stop_stroke_mobjects: list[StrokeMobject] = stop_mobject._stroke_mobjects[:]
        for start_stroke, stop_stroke in it.zip_longest(start_stroke_mobjects, stop_stroke_mobjects, fillvalue=None):
            if start_stroke is None:
                assert stop_stroke is not None
                start_stroke_mobjects.append(stop_stroke.copy().set_style(width=0.0))

            if stop_stroke is None:
                assert start_stroke is not None
                stop_stroke_mobjects.append(start_stroke.copy().set_style(width=0.0))

        intermediate_mobject._stroke_mobjects = [
            stroke_mobject.copy()
            for stroke_mobject in stop_stroke_mobjects
        ]

        def animate_func(alpha_0: Real, alpha: Real) -> None:
            for basedata_descr, interpolate_method in self._SHAPE_INTERPOLATE_METHODS.items():
                if (start_basedata := basedata_descr._get_data(start_mobject)) \
                        is (stop_basedata := basedata_descr._get_data(stop_mobject)):
                    continue
                basedata_descr.__set__(intermediate_mobject, NewData(interpolate_method(
                    start_basedata.data, stop_basedata.data, alpha
                )))

            for start_stroke_mobject, stop_stroke_mobject, intermediate_stroke_mobject in zip(
                start_stroke_mobjects, stop_stroke_mobjects, intermediate_mobject._stroke_mobjects, strict=True
            ):
                for basedata_descr, interpolate_method in self._STROKE_INTERPOLATE_METHODS.items():
                    if (start_basedata := basedata_descr._get_data(start_stroke_mobject)) \
                            is (stop_basedata := basedata_descr._get_data(stop_stroke_mobject)):
                        continue
                    basedata_descr.__set__(intermediate_stroke_mobject, NewData(interpolate_method(
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
