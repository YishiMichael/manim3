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
    LazyUnitaryVariableDescriptor,
    LazyWrapper
)
from ..utils.space import SpaceUtils
from ..utils.shape import (
    MultiLineString,
    Shape
)


_T = TypeVar("_T")
_InstanceT = TypeVar("_InstanceT", bound=LazyObject)
_ElementT = TypeVar("_ElementT", bound=LazyObject)
_DescriptorSetT = TypeVar("_DescriptorSetT")


class VariableInterpolant(Generic[_InstanceT, _ElementT, _DescriptorSetT], ABC):
    __slots__ = (
        "_descriptor",
        "_method"
    )

    def __init__(
        self,
        descriptor: LazyUnitaryVariableDescriptor[_InstanceT, _ElementT, _DescriptorSetT],
        method: Callable[[_ElementT, _ElementT], Callable[[float], _DescriptorSetT]]
    ) -> None:
        super().__init__()
        self._descriptor: LazyUnitaryVariableDescriptor[_InstanceT, _ElementT, _DescriptorSetT] = descriptor  # type checker bug?
        self._method: Callable[[_ElementT, _ElementT], Callable[[float], _DescriptorSetT]] = method

    def _get_intermediate_instance_callback(
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

    @classmethod
    def _get_intermediate_instance_composed_callback(
        cls,
        interpolants: "tuple[VariableInterpolant[_InstanceT, Any, Any], ...]",
        instance_0: _InstanceT,
        instance_1: _InstanceT
    ) -> Callable[[_InstanceT, float], None]:
        callbacks = tuple(
            callback
            for interpolant in interpolants
            if (callback := interpolant._get_intermediate_instance_callback(
                instance_0, instance_1
            )) is not None
        )

        def composed_callback(
            instance: _InstanceT,
            alpha: float
        ) -> None:
            for callback in callbacks:
                callback(instance, alpha)

        return composed_callback


class VariableUnwrappedInterpolant(VariableInterpolant[_InstanceT, LazyWrapper[_T], _DescriptorSetT]):
    def __init__(
        self,
        descriptor: LazyUnitaryVariableDescriptor[_InstanceT, LazyWrapper[_T], _DescriptorSetT],
        method: Callable[[_T, _T], Callable[[float], _DescriptorSetT]]
    ) -> None:

        def new_method(
            variable_0: LazyWrapper[_T],
            variable_1: LazyWrapper[_T]
        ) -> Callable[[float], _DescriptorSetT]:
            return method(variable_0.value, variable_1.value)

        super().__init__(
            descriptor=descriptor,
            method=new_method
        )


class Transform(AlphaAnimation):
    __slots__ = ()

    _SHAPE_INTERPOLANTS: ClassVar[tuple[VariableInterpolant[ShapeMobject, Any, Any], ...]] = (
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
    _STROKE_INTERPOLANTS: ClassVar[tuple[VariableInterpolant[StrokeMobject, Any, Any], ...]] = (
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

        shape_callback = VariableInterpolant._get_intermediate_instance_composed_callback(
            self._SHAPE_INTERPOLANTS, start_mobject, stop_mobject
        )
        stroke_callback_list = [
            VariableInterpolant._get_intermediate_instance_composed_callback(
                self._STROKE_INTERPOLANTS, start_stroke_mobject, stop_stroke_mobject
            )
            for start_stroke_mobject, stop_stroke_mobject in zip(
                start_stroke_mobjects, stop_stroke_mobjects, strict=True
            )
        ]

        def animate_func(
            alpha_0: float,
            alpha: float
        ) -> None:
            shape_callback(intermediate_mobject, alpha)
            for intermediate_stroke_mobject, stroke_callback in zip(
                intermediate_mobject._stroke_mobjects_, stroke_callback_list, strict=True
            ):
                stroke_callback(intermediate_stroke_mobject, alpha)

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
