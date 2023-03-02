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
from ..lazy.core import (
    LazyCollection,
    LazyObjectDescriptor
)
from ..utils.space import SpaceUtils
from ..utils.shape import (
    MultiLineString3D,
    Shape
)


class Transform(AlphaAnimation):
    @staticmethod
    def __shape_interpolate_callback(
        shape_0: Shape,
        shape_1: Shape
    ) -> Callable[[Real], Shape]:
        return shape_0.interpolate_shape_callback(shape_1, has_inlay=True)

    @staticmethod
    def __stroke_interpolate_callback(
        multi_line_string_0: MultiLineString3D,
        multi_line_string_1: MultiLineString3D
    ) -> Callable[[Real], MultiLineString3D]:
        return multi_line_string_0.interpolate_shape_callback(multi_line_string_1, has_inlay=False)

    _SHAPE_INTERPOLATE_CALLBACKS: ClassVar[dict[LazyObjectDescriptor[ShapeMobject, Any], Callable[[Any, Any], Callable[[Real], Any]]]] = {
        ShapeMobject._shape_: __shape_interpolate_callback,
        ShapeMobject._model_matrix_: SpaceUtils.rotational_interpolate_callback,
        ShapeMobject._color_: SpaceUtils.lerp_callback,
        ShapeMobject._opacity_: SpaceUtils.lerp_callback,
        ShapeMobject._ambient_strength_: SpaceUtils.lerp_callback,
        ShapeMobject._specular_strength_: SpaceUtils.lerp_callback,
        ShapeMobject._shininess_: SpaceUtils.lerp_callback
    }

    _STROKE_INTERPOLATE_CALLBACKS: ClassVar[dict[LazyObjectDescriptor[StrokeMobject, Any], Callable[[Any, Any], Callable[[Real], Any]]]] = {
        StrokeMobject._multi_line_string_3d_: __stroke_interpolate_callback,
        StrokeMobject._model_matrix_: SpaceUtils.rotational_interpolate_callback,
        StrokeMobject._color_: SpaceUtils.lerp_callback,
        StrokeMobject._opacity_: SpaceUtils.lerp_callback,
        StrokeMobject._width_: SpaceUtils.lerp_callback,
        StrokeMobject._color_: SpaceUtils.lerp_callback,
        StrokeMobject._opacity_: SpaceUtils.lerp_callback,
        StrokeMobject._dilate_: SpaceUtils.lerp_callback
    }

    def __init__(
        self,
        start_mobject: ShapeMobject,
        stop_mobject: ShapeMobject,
        *,
        run_time: Real = 2.0,
        rate_func: Callable[[Real], Real] | None = None
    ) -> None:
        intermediate_mobject = stop_mobject.copy()

        start_stroke_mobjects: LazyCollection[StrokeMobject] = start_mobject._stroke_mobjects_
        stop_stroke_mobjects: LazyCollection[StrokeMobject] = stop_mobject._stroke_mobjects_
        for start_stroke, stop_stroke in it.zip_longest(start_stroke_mobjects, stop_stroke_mobjects, fillvalue=None):
            if start_stroke is None:
                assert stop_stroke is not None
                start_stroke_mobjects.add(stop_stroke.copy().set_style(width=0.0))

            if stop_stroke is None:
                assert start_stroke is not None
                stop_stroke_mobjects.add(start_stroke.copy().set_style(width=0.0))

        #intermediate_mobject._stroke_mobjects_.add(
        #    stroke_mobject.copy()
        #    for stroke_mobject in stop_stroke_mobjects
        #)

        shape_callbacks = {
            variable_descr: interpolate_method(start_variable.value, stop_variable.value)
            for variable_descr, interpolate_method in self._SHAPE_INTERPOLATE_CALLBACKS.items()
            if (start_variable := variable_descr.__get__(start_mobject)) \
                is not (stop_variable := variable_descr.__get__(stop_mobject))
        }
        stroke_callbacks_list = [
            {
                variable_descr: interpolate_method(start_variable.value, stop_variable.value)
                for variable_descr, interpolate_method in self._STROKE_INTERPOLATE_CALLBACKS.items()
                if (start_variable := variable_descr.__get__(start_stroke_mobject)) \
                    is not (stop_variable := variable_descr.__get__(stop_stroke_mobject))
            }
            for start_stroke_mobject, stop_stroke_mobject in zip(
                start_stroke_mobjects, stop_stroke_mobjects, strict=True
            )
        ]

        def animate_func(
            alpha_0: Real,
            alpha: Real
        ) -> None:
            for variable_descr, callback in shape_callbacks.items():
                variable_descr.__set__(intermediate_mobject, callback(alpha))

            for callbacks, intermediate_stroke_mobject in zip(
                stroke_callbacks_list, intermediate_mobject._stroke_mobjects_, strict=True
            ):
                for variable_descr, callback in callbacks.items():
                    variable_descr.__set__(intermediate_stroke_mobject, callback(alpha))

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
