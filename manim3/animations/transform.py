__all__ = ["Transform"]


import itertools as it
from typing import (
    Any,
    Callable,
    ClassVar
)

from ..animations.animation import AlphaAnimation
from ..custom_typing import (
    FloatsT,
    Mat3T,
    Mat4T,
    Vec2T,
    Vec2sT,
    Vec3T,
    Vec3sT,
    Vec4T,
    Vec4sT
)
from ..mobjects.shape_mobject import ShapeMobject
from ..mobjects.stroke_mobject import StrokeMobject
from ..lazy.core import LazyObjectVariableDescriptor
from ..lazy.interface import LazyWrapper
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
    ) -> Callable[[float], Shape]:
        return shape_0.interpolate_shape_callback(shape_1, has_inlay=True)

    @staticmethod
    def __stroke_interpolate_callback(
        multi_line_string_0: MultiLineString3D,
        multi_line_string_1: MultiLineString3D
    ) -> Callable[[float], MultiLineString3D]:
        return multi_line_string_0.interpolate_shape_callback(multi_line_string_1, has_inlay=False)

    @staticmethod
    def __rotational_interpolate_callback(
        matrix_0: LazyWrapper[Mat4T],
        matrix_1: LazyWrapper[Mat4T]
    ) -> Callable[[float], Mat4T]:
        return SpaceUtils.rotational_interpolate_callback(matrix_0.value, matrix_1.value)

    @staticmethod
    def __lerp_callback(
        tensor_0: LazyWrapper[float | FloatsT | Vec2T | Vec2sT | Vec3T | Vec3sT | Vec4T | Vec4sT | Mat3T | Mat4T],
        tensor_1: LazyWrapper[float | FloatsT | Vec2T | Vec2sT | Vec3T | Vec3sT | Vec4T | Vec4sT | Mat3T | Mat4T]
    ) -> Callable[[float], float | FloatsT | Vec2T | Vec2sT | Vec3T | Vec3sT | Vec4T | Vec4sT | Mat3T | Mat4T]:
        return SpaceUtils.lerp_callback(tensor_0.value, tensor_1.value)

    _SHAPE_INTERPOLATE_CALLBACKS: ClassVar[dict[LazyObjectVariableDescriptor[ShapeMobject, Any], Callable[[Any, Any], Callable[[float], Any]]]] = {
        ShapeMobject._shape_: __shape_interpolate_callback,
        ShapeMobject._model_matrix_: __rotational_interpolate_callback,
        ShapeMobject._color_: __lerp_callback,
        ShapeMobject._opacity_: __lerp_callback,
        ShapeMobject._ambient_strength_: __lerp_callback,
        ShapeMobject._specular_strength_: __lerp_callback,
        ShapeMobject._shininess_: __lerp_callback
    }

    _STROKE_INTERPOLATE_CALLBACKS: ClassVar[dict[LazyObjectVariableDescriptor[StrokeMobject, Any], Callable[[Any, Any], Callable[[float], Any]]]] = {
        StrokeMobject._multi_line_string_3d_: __stroke_interpolate_callback,
        StrokeMobject._model_matrix_: __rotational_interpolate_callback,
        StrokeMobject._color_: __lerp_callback,
        StrokeMobject._opacity_: __lerp_callback,
        StrokeMobject._width_: __lerp_callback,
        StrokeMobject._color_: __lerp_callback,
        StrokeMobject._opacity_: __lerp_callback,
        StrokeMobject._dilate_: __lerp_callback
    }

    def __init__(
        self,
        start_mobject: ShapeMobject,
        stop_mobject: ShapeMobject,
        *,
        run_time: float = 2.0,
        rate_func: Callable[[float], float] | None = None
    ) -> None:
        intermediate_mobject = stop_mobject.copy()

        start_stroke_mobjects = list(start_mobject._stroke_mobjects_)
        stop_stroke_mobjects = list(stop_mobject._stroke_mobjects_)
        for start_stroke, stop_stroke in it.zip_longest(start_stroke_mobjects, stop_stroke_mobjects, fillvalue=None):
            if start_stroke is None:
                assert stop_stroke is not None
                start_stroke = stop_stroke.copy().set_style(width=0.0)
                start_stroke._model_matrix_ = start_mobject._model_matrix_
                start_stroke._multi_line_string_3d_ = start_mobject._shape_._multi_line_string_3d_  # TODO
                start_stroke_mobjects.append(start_stroke)

            if stop_stroke is None:
                assert start_stroke is not None
                stop_stroke = start_stroke.copy().set_style(width=0.0)
                stop_stroke._model_matrix_ = stop_mobject._model_matrix_
                stop_stroke._multi_line_string_3d_ = stop_mobject._shape_._multi_line_string_3d_
                stop_stroke_mobjects.append(stop_stroke)
                intermediate_mobject._stroke_mobjects_.add(stop_stroke)

        shape_callbacks = {
            variable_descriptor: interpolate_method(start_variable, stop_variable)
            for variable_descriptor, interpolate_method in self._SHAPE_INTERPOLATE_CALLBACKS.items()
            if (start_variable := variable_descriptor.__get__(start_mobject)) \
                is not (stop_variable := variable_descriptor.__get__(stop_mobject))
        }
        stroke_callbacks_list = [
            {
                variable_descriptor: interpolate_method(start_variable, stop_variable)
                for variable_descriptor, interpolate_method in self._STROKE_INTERPOLATE_CALLBACKS.items()
                if (start_variable := variable_descriptor.__get__(start_stroke_mobject)) \
                    is not (stop_variable := variable_descriptor.__get__(stop_stroke_mobject))
            }
            for start_stroke_mobject, stop_stroke_mobject in zip(
                start_stroke_mobjects, stop_stroke_mobjects, strict=True
            )
        ]

        def animate_func(
            alpha_0: float,
            alpha: float
        ) -> None:
            for variable_descriptor, callback in shape_callbacks.items():
                variable_descriptor.__set__(intermediate_mobject, callback(alpha))

            for callbacks, intermediate_stroke_mobject in zip(
                stroke_callbacks_list, intermediate_mobject._stroke_mobjects_, strict=True
            ):
                for variable_descriptor, callback in callbacks.items():
                    variable_descriptor.__set__(intermediate_stroke_mobject, callback(alpha))

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
