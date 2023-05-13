from typing import Callable

from ..animations.animation import Animation
from ..custom_typing import TimelineT
#from ..lazy.lazy import (
#    LazyContainer,
#    LazyObject,
#    LazyVariableDescriptor
#)
#from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.mobject import (
    Mobject,
    MobjectMeta
)
#from ..mobjects.shape_mobject import ShapeMobject
#from ..mobjects.stroke_mobject import StrokeMobject
#from ..shape.line_string import MultiLineString
#from ..shape.shape import Shape
from ..utils.rate import RateUtils
#from ..utils.space import SpaceUtils


#_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
#_DescriptorSetT = TypeVar("_DescriptorSetT")
#_DescriptorRawT = TypeVar("_DescriptorRawT")


class Transform(Animation):
    __slots__ = (
        "_start_mobject",
        "_stop_mobject",
        "_intermediate_mobject"
    )

    #_interpolate_methods: ClassVar[dict[LazyVariableDescriptor, Callable[[Any, Any], Callable[[float], Any]]]] = {
    #    Mobject._children_: NotImplemented,
    #    Mobject._real_descendants_: NotImplemented,
    #    Mobject._camera_: NotImplemented,
    #    MeshMobject._lighting_: NotImplemented,
    #    Mobject._model_matrix_: SpaceUtils.lerp,
    #    Mobject._color_: SpaceUtils.lerp,
    #    Mobject._opacity_: SpaceUtils.lerp,
    #    MeshMobject._ambient_strength_: SpaceUtils.lerp,
    #    MeshMobject._specular_strength_: SpaceUtils.lerp,
    #    MeshMobject._shininess_: SpaceUtils.lerp,
    #    ShapeMobject._shape_: Shape.get_interpolator,
    #    StrokeMobject._multi_line_string_: MultiLineString.get_interpolator,
    #    StrokeMobject._width_: SpaceUtils.lerp,
    #    StrokeMobject._dilate_: SpaceUtils.lerp
    #}

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        intermediate_mobject: Mobject = Mobject()
        callbacks: list[Callable[[float], None]] = []
        # Requires descendants of `start_mobject` and `stop_mobject` perfectly aligned.
        for descendant_0, descendant_1 in zip(
            start_mobject.iter_descendants(),
            stop_mobject.iter_descendants(),
            strict=True
        ):
            intermediate_descendant = descendant_0.copy_standalone()
            intermediate_mobject.add(intermediate_descendant)
            callbacks.append(MobjectMeta._interpolate(descendant_0, descendant_1)(intermediate_descendant))

        #intermediate_mobjects_with_callback = list(
        #    (
        #        descendant_mobject := mobject_0.copy_standalone(),
        #        Mobject._interpolate(mobject_0, mobject_1)(descendant_mobject)
        #    )
        #    for mobject_0, mobject_1 in zip(
        #        start_mobject.iter_descendants(),
        #        stop_mobject.iter_descendants(),
        #        strict=True
        #    )
        #)

        def updater(
            alpha: float
        ) -> None:
            for callback in callbacks:
                callback(alpha)

        super().__init__(
            run_time=run_time,
            relative_rate=RateUtils.adjust(rate_func, run_time_scale=run_time),
            updater=updater
        )
        self._start_mobject: Mobject = start_mobject
        self._stop_mobject: Mobject = stop_mobject
        self._intermediate_mobject: Mobject = intermediate_mobject

    def timeline(self) -> TimelineT:
        start_mobject = self._start_mobject
        stop_mobject = self._stop_mobject
        intermediate_mobject = self._intermediate_mobject
        parents = list(start_mobject.iter_parents())
        start_mobject.discarded_by(*parents)
        intermediate_mobject.added_by(*parents)
        yield from self.wait()
        intermediate_mobject.discarded_by(*parents)
        stop_mobject.added_by(*parents)

    #@classmethod
    #def _get_interpolate_callback(
    #    cls,
    #    instance_0: _InstanceT,
    #    instance_1: _InstanceT
    #) -> Callable[[_InstanceT, float], None]:
    #    # Two objects are said to be "interpolable" if their lazy variable descriptors match
    #    # (a weaker condition than being of the same type), and their lazy variables without
    #    # `interpolate_method` specified should match in values.
    #    assert (lazy_descriptors := type(instance_0)._lazy_variable_descriptors) == type(instance_1)._lazy_variable_descriptors
    #    descriptor_callbacks = [
    #        descriptor_callback
    #        for descriptor in lazy_descriptors
    #        if (descriptor_callback := cls._get_interpolate_descriptor_callback(
    #            descriptor, cls._interpolate_methods.get(descriptor), instance_0, instance_1
    #        )) is not None
    #    ]

    #    def callback(
    #        dst: _InstanceT,
    #        alpha: float
    #    ) -> None:
    #        for descriptor_callback in descriptor_callbacks:
    #            descriptor_callback(dst, alpha)

    #    return callback

    #@classmethod
    #def _get_interpolate_descriptor_callback(
    #    cls,
    #    descriptor: LazyVariableDescriptor[_InstanceT, LazyContainer, Any, _DescriptorSetT, _DescriptorRawT],
    #    interpolate_method: Callable[[_DescriptorRawT, _DescriptorRawT], Callable[[float], _DescriptorSetT]] | None,
    #    instance_0: _InstanceT,
    #    instance_1: _InstanceT
    #) -> Callable[[_InstanceT, float], None] | None:
    #    if interpolate_method is NotImplemented:
    #        # Ignore variables with `interpolate_method` explicitly assigned to `NotImplemented`.
    #        return None

    #    container_0 = descriptor.get_container(instance_0)
    #    container_1 = descriptor.get_container(instance_1)
    #    if container_0._match_elements(container_1):
    #        return None

    #    if interpolate_method is None:
    #        raise ValueError
    #    interpolator = interpolate_method(
    #        descriptor.converter.convert_rget(container_0),
    #        descriptor.converter.convert_rget(container_1)
    #    )

    #    def callback(
    #        dst: _InstanceT,
    #        alpha: float
    #    ) -> None:
    #        new_container = descriptor.converter.convert_set(interpolator(alpha))
    #        descriptor.set_container(dst, new_container)

    #    return callback
