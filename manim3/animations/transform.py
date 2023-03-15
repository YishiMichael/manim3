__all__ = ["Transform"]


from abc import ABC
import itertools as it
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    TypeVar
)

from ..animations.animation import AlphaAnimation
from ..mobjects.shape_mobject import ShapeMobject
from ..mobjects.stroke_mobject import StrokeMobject
from ..lazy.core import (
    LazyObject,
    LazyObjectVariableDescriptor,
    LazyWrapper
)
from ..utils.space import SpaceUtils
from ..utils.shape import (
    MultiLineString,
    Shape
)


_T = TypeVar("_T")
_InputT = TypeVar("_InputT")
_InstanceT = TypeVar("_InstanceT", bound=LazyObject)
_LazyObjectT = TypeVar("_LazyObjectT", bound=LazyObject)


class VariableInterpolant(Generic[_InstanceT, _LazyObjectT], ABC):
    __slots__ = (
        "_descriptor",
        "_method"
    )

    def __init__(
        self,
        descriptor: LazyObjectVariableDescriptor[_InstanceT, _LazyObjectT, _InputT],
        method: Callable[[_LazyObjectT, _LazyObjectT], Callable[[float], _InputT]]
    ) -> None:
        super().__init__()
        self._descriptor: LazyObjectVariableDescriptor[_InstanceT, _LazyObjectT, _InputT] = descriptor  # type checker bug?
        self._method: Callable[[_LazyObjectT, _LazyObjectT], Callable[[float], _InputT]] = method

    def get_intermediate_instance_callback(
        self,
        instance_0: _InstanceT,
        instance_1: _InstanceT
    ) -> Callable[[_InstanceT, float], None] | None:
        descriptor = self._descriptor
        variable_0 = descriptor.__get__(instance_0)
        variable_1 = descriptor.__get__(instance_1)
        if variable_0 is variable_1:
            return None

        callback = self._method(variable_0, variable_1)

        def intermediate_instance_callback(
            instance: _InstanceT,
            alpha: float
        ) -> None:
            if variable_0 is variable_1:
                return
            descriptor.__set__(instance, callback(alpha))

        return intermediate_instance_callback


class VariableUnwrappedInterpolant(VariableInterpolant[_InstanceT, LazyWrapper[_T]]):
    def __init__(
        self,
        descriptor: LazyObjectVariableDescriptor[_InstanceT, LazyWrapper[_T], _InputT],
        method: Callable[[_T, _T], Callable[[float], _InputT]]
    ) -> None:

        def new_method(
            variable_0: LazyWrapper[_T],
            variable_1: LazyWrapper[_T]
        ) -> Callable[[float], _InputT]:
            return method(variable_0.value, variable_1.value)

        super().__init__(
            descriptor=descriptor,
            method=new_method
        )


class Transform(AlphaAnimation):
    __slots__ = ()

    #@staticmethod
    #def __shape_interpolate_callback(
    #    shape_0: Shape,
    #    shape_1: Shape
    #) -> Callable[[float], Shape]:
    #    return shape_0.interpolate_shape_callback(shape_1, has_inlay=True)

    #@staticmethod
    #def __stroke_interpolate_callback(
    #    multi_line_string_0: MultiLineString,
    #    multi_line_string_1: MultiLineString
    #) -> Callable[[float], MultiLineString]:
    #    return multi_line_string_0.interpolate_shape_callback(multi_line_string_1, has_inlay=False)

    #@staticmethod
    #def __rotational_interpolate_callback(
    #    matrix_0: LazyWrapper[Mat4T],
    #    matrix_1: LazyWrapper[Mat4T]
    #) -> Callable[[float], Mat4T]:
    #    return SpaceUtils.rotational_interpolate_callback(matrix_0.value, matrix_1.value)

    #@staticmethod
    #def __lerp_callback(
    #    tensor_0: LazyWrapper[float | FloatsT | Vec2T | Vec2sT | Vec3T | Vec3sT | Vec4T | Vec4sT | Mat3T | Mat4T],
    #    tensor_1: LazyWrapper[float | FloatsT | Vec2T | Vec2sT | Vec3T | Vec3sT | Vec4T | Vec4sT | Mat3T | Mat4T]
    #) -> Callable[[float], float | FloatsT | Vec2T | Vec2sT | Vec3T | Vec3sT | Vec4T | Vec4sT | Mat3T | Mat4T]:
    #    return SpaceUtils.lerp_callback(tensor_0.value, tensor_1.value)

    _SHAPE_INTERPOLANTS: ClassVar[tuple[VariableInterpolant[ShapeMobject, Any], ...]] = (
        VariableInterpolant(
            descriptor=ShapeMobject._shape_,
            method=Shape.interpolate_shape_callback
        ),
        VariableUnwrappedInterpolant(
            descriptor=ShapeMobject._model_matrix_,
            method=SpaceUtils.rotational_interpolate_callback
        ),
        VariableUnwrappedInterpolant(
            descriptor=ShapeMobject._color_,
            method=SpaceUtils.lerp_callback
        ),
        VariableUnwrappedInterpolant(
            descriptor=ShapeMobject._opacity_,
            method=SpaceUtils.lerp_callback
        ),
        VariableUnwrappedInterpolant(
            descriptor=ShapeMobject._ambient_strength_,
            method=SpaceUtils.lerp_callback
        ),
        VariableUnwrappedInterpolant(
            descriptor=ShapeMobject._specular_strength_,
            method=SpaceUtils.lerp_callback
        ),
        VariableUnwrappedInterpolant(
            descriptor=ShapeMobject._shininess_,
            method=SpaceUtils.lerp_callback
        )
    )

    #_STROKE_INTERPOLATE_CALLBACKS: ClassVar[dict[LazyObjectVariableDescriptor[StrokeMobject, Any], Callable[[Any, Any], Callable[[float], Any]]]] = {
    #    StrokeMobject._multi_line_string_: __stroke_interpolate_callback,
    #    StrokeMobject._model_matrix_: __rotational_interpolate_callback,
    #    StrokeMobject._color_: __lerp_callback,
    #    StrokeMobject._opacity_: __lerp_callback,
    #    StrokeMobject._width_: __lerp_callback,
    #    StrokeMobject._color_: __lerp_callback,
    #    StrokeMobject._opacity_: __lerp_callback,
    #    StrokeMobject._dilate_: __lerp_callback
    #}

    _STROKE_INTERPOLANTS: ClassVar[tuple[VariableInterpolant[StrokeMobject, Any], ...]] = (
        VariableInterpolant(
            descriptor=StrokeMobject._multi_line_string_,
            method=MultiLineString.interpolate_shape_callback
        ),
        VariableUnwrappedInterpolant(
            descriptor=StrokeMobject._model_matrix_,
            method=SpaceUtils.rotational_interpolate_callback
        ),
        VariableUnwrappedInterpolant(
            descriptor=StrokeMobject._color_,
            method=SpaceUtils.lerp_callback
        ),
        VariableUnwrappedInterpolant(
            descriptor=StrokeMobject._opacity_,
            method=SpaceUtils.lerp_callback
        ),
        VariableUnwrappedInterpolant(
            descriptor=StrokeMobject._width_,
            method=SpaceUtils.lerp_callback
        ),
        VariableUnwrappedInterpolant(
            descriptor=StrokeMobject._dilate_,
            method=SpaceUtils.lerp_callback
        )
    )

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
                start_mobject.adjust_stroke_shape(start_stroke)
                start_stroke_mobjects.append(start_stroke)

            if stop_stroke is None:
                assert start_stroke is not None
                stop_stroke = start_stroke.copy().set_style(width=0.0)
                stop_mobject.adjust_stroke_shape(stop_stroke)
                stop_stroke_mobjects.append(stop_stroke)
                intermediate_mobject.add_stroke_mobject(stop_stroke)

        shape_callbacks = [
            callback
            for interpolant in self._SHAPE_INTERPOLANTS
            if (callback := interpolant.get_intermediate_instance_callback(
                start_mobject, stop_mobject
            )) is not None
        ]
        stroke_callbacks_list = [
            [
                callback
                for interpolant in self._STROKE_INTERPOLANTS
                if (callback := interpolant.get_intermediate_instance_callback(
                    start_stroke_mobject, stop_stroke_mobject
                )) is not None
            ]
            for start_stroke_mobject, stop_stroke_mobject in zip(
                start_stroke_mobjects, stop_stroke_mobjects, strict=True
            )
        ]

        def animate_func(
            alpha_0: float,
            alpha: float
        ) -> None:
            for shape_callback in shape_callbacks:
                shape_callback(intermediate_mobject, alpha)
            for intermediate_stroke_mobject, stroke_callbacks in zip(
                intermediate_mobject._stroke_mobjects_, stroke_callbacks_list, strict=True
            ):
                for stroke_callback in stroke_callbacks:
                    stroke_callback(intermediate_stroke_mobject, alpha)
            #for descriptor, callback in shape_callbacks.items():
            #    descriptor.__set__(intermediate_mobject, callback(alpha))

            #for callbacks, intermediate_stroke_mobject in zip(
            #    stroke_callbacks_list, intermediate_mobject._stroke_mobjects_, strict=True
            #):
            #    for descriptor, callback in callbacks.items():
            #        descriptor.__set__(intermediate_stroke_mobject, callback(alpha))

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
