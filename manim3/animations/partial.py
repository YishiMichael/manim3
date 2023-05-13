from typing import (
    Any,
    Callable,
    ClassVar,
    TypeVar
)

import numpy as np

from ..animations.animation import Animation
from ..custom_typing import TimelineT
from ..lazy.lazy import (
    LazyContainer,
    LazyObject,
    LazyVariableDescriptor
)
from ..mobjects.mobject import Mobject
from ..mobjects.shape_mobject import ShapeMobject
from ..mobjects.stroke_mobject import StrokeMobject
from ..shape.line_string import MultiLineString
from ..shape.shape import Shape
from ..utils.rate import RateUtils


_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
_DescriptorSetT = TypeVar("_DescriptorSetT")
_DescriptorRGetT = TypeVar("_DescriptorRGetT")


class PartialAnimation(Animation):
    __slots__ = ("_mobject",)

    _partial_methods: ClassVar[dict[LazyVariableDescriptor, Callable[[Any], Callable[[float, float], Any]]]] = {
        ShapeMobject._shape_: Shape.get_partialor,
        StrokeMobject._multi_line_string_: MultiLineString.get_partialor
    }

    def __init__(
        self,
        mobject: Mobject,
        alpha_to_boundary_values: Callable[[float], tuple[float, float]],
        *,
        backwards: bool = False,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        descendant_mobjects_with_callback = list(
            (
                descendant,
                self._get_partial_callback(descendant, alpha_to_boundary_values, backwards)
            )
            for descendant in mobject.iter_descendants()
        )

        def updater(
            alpha: float
        ) -> None:
            for mobject, callback in descendant_mobjects_with_callback:
                callback(mobject, alpha)

        super().__init__(
            run_time=run_time,
            relative_rate=RateUtils.adjust(rate_func, run_time_scale=run_time),
            updater=updater
        )
        self._mobject: Mobject = mobject

    def timeline(self) -> TimelineT:
        yield from self.wait()

    @classmethod
    def _get_partial_callback(
        cls,
        instance: _InstanceT,
        alpha_to_boundary_values: Callable[[float], tuple[float, float]],
        backwards: bool
    ) -> Callable[[_InstanceT, float], None]:
        descriptor_callbacks = [
            descriptor_callback
            for descriptor in type(instance)._lazy_variable_descriptors
            if (descriptor_callback := cls._get_partial_descriptor_callback(
                descriptor, cls._partial_methods.get(descriptor), instance
            )) is not None
        ]

        def callback(
            dst: _InstanceT,
            alpha: float
        ) -> None:
            start, stop = alpha_to_boundary_values(alpha)
            if backwards:
                start, stop = 1.0 - stop, 1.0 - start
            for descriptor_callback in descriptor_callbacks:
                descriptor_callback(dst, start, stop)

        return callback

    @classmethod
    def _get_partial_descriptor_callback(
        cls,
        descriptor: LazyVariableDescriptor[_InstanceT, LazyContainer, Any, _DescriptorSetT, _DescriptorRGetT],
        partial_method: Callable[[_DescriptorRGetT], Callable[[float, float], _DescriptorSetT]] | None,
        instance: _InstanceT
    ) -> Callable[[_InstanceT, float, float], None] | None:
        if partial_method is None:
            return None

        container = descriptor.get_container(instance)
        partialor = partial_method(
            descriptor.converter.convert_rget(container)
        )

        def callback(
            dst: _InstanceT,
            start: float,
            stop: float
        ) -> None:
            new_container = descriptor.converter.convert_set(partialor(start, stop))
            descriptor.set_container(dst, new_container)

        return callback


class PartialCreate(PartialAnimation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        backwards: bool = False,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:

        def alpha_to_boundary_values(
            alpha: float
        ) -> tuple[float, float]:
            return (0.0, alpha)

        super().__init__(
            mobject=mobject,
            alpha_to_boundary_values=alpha_to_boundary_values,
            backwards=backwards,
            run_time=run_time,
            rate_func=rate_func
        )


class PartialUncreate(PartialAnimation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        backwards: bool = False,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:

        def alpha_to_boundary_values(
            alpha: float
        ) -> tuple[float, float]:
            return (0.0, 1.0 - alpha)

        super().__init__(
            mobject=mobject,
            alpha_to_boundary_values=alpha_to_boundary_values,
            backwards=backwards,
            run_time=run_time,
            rate_func=rate_func
        )

    def timeline(self) -> TimelineT:
        yield from super().timeline()
        self._mobject.discarded_by(*self._mobject.iter_parents())


class PartialFlash(PartialAnimation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        flash_proportion: float = 1.0 / 16,
        backwards: bool = False,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        assert flash_proportion >= 0.0

        def clip_proportion(
            alpha: float
        ) -> float:
            return float(np.clip(alpha, 0.0, 1.0))

        def alpha_to_boundary_values(
            alpha: float
        ) -> tuple[float, float]:
            return (
                clip_proportion(alpha * (1.0 + flash_proportion) - flash_proportion),
                clip_proportion(alpha * (1.0 + flash_proportion))
            )

        super().__init__(
            mobject=mobject,
            alpha_to_boundary_values=alpha_to_boundary_values,
            backwards=backwards,
            run_time=run_time,
            rate_func=rate_func
        )

    def timeline(self) -> TimelineT:
        yield from super().timeline()
        self._mobject.discarded_by(*self._mobject.iter_parents())
