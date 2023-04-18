__all__ = ["Transform"]


from functools import lru_cache
import itertools as it
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    TypeVar
)

from ..animations.animation import (
    Animation,
    RegroupItem,
    RegroupVerb
)
from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.mobject import Mobject
from ..mobjects.shape_mobject import ShapeMobject
from ..mobjects.stroke_mobject import StrokeMobject
from ..lazy.core import (
    LazyContainer,
    LazyObject,
    LazyVariableDescriptor,
    LazyWrapper
)
from ..utils.rate import RateUtils
from ..utils.space import SpaceUtils
from ..utils.shape import (
    MultiLineString,
    Shape
)


_T = TypeVar("_T")
_ContainerT = TypeVar("_ContainerT", bound=LazyContainer)
_ElementT = TypeVar("_ElementT", bound=LazyObject)
_InstanceT = TypeVar("_InstanceT", bound=LazyObject)
_DescriptorGetT = TypeVar("_DescriptorGetT")
_DescriptorSetT = TypeVar("_DescriptorSetT")
_MobjectT = TypeVar("_MobjectT", bound=Mobject)


class VariableInterpolant(Generic[_InstanceT, _ContainerT, _ElementT, _DescriptorGetT, _DescriptorSetT]):
    __slots__ = (
        "_descriptor",
        "_method"
    )

    def __init__(
        self,
        descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, _ElementT, _DescriptorGetT, _DescriptorSetT],
        method: Callable[[_DescriptorGetT, _DescriptorGetT], Callable[[float], _DescriptorSetT]]
    ) -> None:
        super().__init__()
        self._descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, _ElementT, _DescriptorGetT, _DescriptorSetT] = descriptor
        self._method: Callable[[_DescriptorGetT, _DescriptorGetT], Callable[[float], _DescriptorSetT]] = method

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
            descriptor.__set__(instance, callback(alpha))

        return intermediate_instance_callback

    @classmethod
    def _get_intermediate_instance_composed_callback(
        cls,
        interpolants: "tuple[VariableInterpolant[_InstanceT, Any, Any, Any, Any], ...]",
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

    @classmethod
    def _wrap_method(
        cls,
        method: Callable[[_T, _T], Callable[[float], _DescriptorSetT]]
    ) -> Callable[[LazyWrapper[_T], LazyWrapper[_T]], Callable[[float], _DescriptorSetT]]:

        def new_method(
            variable_0: LazyWrapper[_T],
            variable_1: LazyWrapper[_T]
        ) -> Callable[[float], _DescriptorSetT]:
            return method(variable_0.value, variable_1.value)

        return new_method


class Transform(Animation):
    __slots__ = ()

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject,
        *,
        run_time: float = 2.0,
        rate_func: Callable[[float], float] | None = None,
        replace: bool = False
    ) -> None:
        intermediate_mobjects_with_callback: list[tuple[Mobject, Callable[[Mobject, float], None]]] = [
            (mobject_0.copy(broadcast=False), VariableInterpolant._get_intermediate_instance_composed_callback(
                interpolants=self._get_class_interpolants(mobject_cls),
                instance_0=mobject_0,
                instance_1=mobject_1
            ))
            for mobject_cls, mobject_0, mobject_1 in self._zip_mobjects_by_class(
                start_mobject.iter_descendants(),
                stop_mobject.iter_descendants()
            )
        ]

        def alpha_animate_func(
            alpha_0: float,
            alpha: float
        ) -> None:
            for mobject, callback in intermediate_mobjects_with_callback:
                callback(mobject, alpha)

        intermediate_mobject = Mobject().add(*(mobject for mobject, _ in intermediate_mobjects_with_callback))
        all_parents = [
            *start_mobject.iter_parents(),
            *stop_mobject.iter_parents()
        ]  # TODO
        alpha_regroup_items = [
            (0.0, RegroupItem(
                mobjects=all_parents,
                verb=RegroupVerb.DISCARD,
                targets=start_mobject
            )),
            (0.0, RegroupItem(
                mobjects=all_parents,
                verb=RegroupVerb.ADD,
                targets=intermediate_mobject
            )),
            (1.0, RegroupItem(
                mobjects=all_parents,
                verb=RegroupVerb.DISCARD,
                targets=intermediate_mobject
            )),
            (1.0, RegroupItem(
                mobjects=all_parents,
                verb=RegroupVerb.ADD,
                targets=stop_mobject if replace else start_mobject
            ))
        ]
        if not replace:
            alpha_regroup_items.append((1.0, RegroupItem(
                mobjects=start_mobject,
                verb=RegroupVerb.BECOMES,
                targets=stop_mobject
            )))

        if rate_func is None:
            rate_func = RateUtils.smooth
        super().__init__(
            alpha_animate_func=alpha_animate_func,
            alpha_regroup_items=alpha_regroup_items,
            start_time=0.0,
            stop_time=run_time,
            rate_func=RateUtils.compose(rate_func, lambda t: t / run_time)
        )

    @classmethod
    def _zip_mobjects_by_class(
        cls,
        mobjects_0: Iterable[Mobject],
        mobjects_1: Iterable[Mobject]
    ) -> Generator[tuple[type[Mobject], Mobject, Mobject], None, None]:

        def get_class(
            mobject: Mobject
        ) -> type[Mobject]:
            for mobject_cls in (
                ShapeMobject,
                StrokeMobject,
                MeshMobject,
                Mobject
            ):
                if isinstance(mobject, mobject_cls):
                    return mobject_cls
            raise TypeError

        def get_placeholder_mobject(
            mobject: _MobjectT
        ) -> _MobjectT:
            result = mobject.copy(broadcast=False)
            if isinstance(result, MeshMobject):
                result.set_style(opacity=0.0, is_transparent=mobject._is_transparent_.value)  # TODO
            elif isinstance(result, StrokeMobject):
                result.set_style(width=0.0)
            return result

        # Why `list()` is needed here?
        for mobject_0, mobject_1 in it.zip_longest(list(mobjects_0), list(mobjects_1), fillvalue=None):
            if mobject_0 is not None and mobject_1 is None:
                mobject_1 = get_placeholder_mobject(mobject_0)
            elif mobject_0 is None and mobject_1 is not None:
                mobject_0 = get_placeholder_mobject(mobject_1)
            if mobject_0 is None or mobject_1 is None:
                raise ValueError  # never

            cls_0 = get_class(mobject_0)
            cls_1 = get_class(mobject_1)
            if cls_0 is cls_1:
                yield cls_0, mobject_0, mobject_1
            else:
                yield cls_0, mobject_0, get_placeholder_mobject(mobject_0)
                yield cls_1, get_placeholder_mobject(mobject_1), mobject_1

    @classmethod
    @lru_cache(maxsize=16)
    def _get_class_interpolants(
        cls,
        mobject_cls: type[_MobjectT]
    ) -> tuple[VariableInterpolant[_MobjectT, Any, Any, Any, Any], ...]:
        class_specialized_interpolants_dict = {
            Mobject: [
                VariableInterpolant(
                    descriptor=Mobject._model_matrix_,
                    method=VariableInterpolant._wrap_method(SpaceUtils.rotational_interpolate_callback)
                )
            ],
            MeshMobject: [
                VariableInterpolant(
                    descriptor=MeshMobject._color_,
                    method=VariableInterpolant._wrap_method(SpaceUtils.lerp_callback)
                ),
                VariableInterpolant(
                    descriptor=MeshMobject._opacity_,
                    method=VariableInterpolant._wrap_method(SpaceUtils.lerp_callback)
                ),
                VariableInterpolant(
                    descriptor=MeshMobject._ambient_strength_,
                    method=VariableInterpolant._wrap_method(SpaceUtils.lerp_callback)
                ),
                VariableInterpolant(
                    descriptor=MeshMobject._specular_strength_,
                    method=VariableInterpolant._wrap_method(SpaceUtils.lerp_callback)
                ),
                VariableInterpolant(
                    descriptor=MeshMobject._shininess_,
                    method=VariableInterpolant._wrap_method(SpaceUtils.lerp_callback)
                )
            ],
            ShapeMobject: [
                VariableInterpolant(
                    descriptor=ShapeMobject._shape_,
                    method=Shape.interpolate_shape_callback
                )
            ],
            StrokeMobject: [
                VariableInterpolant(
                    descriptor=StrokeMobject._multi_line_string_,
                    method=MultiLineString.interpolate_shape_callback,
                ),
                VariableInterpolant(
                    descriptor=StrokeMobject._color_,
                    method=VariableInterpolant._wrap_method(SpaceUtils.lerp_callback)
                ),
                VariableInterpolant(
                    descriptor=StrokeMobject._opacity_,
                    method=VariableInterpolant._wrap_method(SpaceUtils.lerp_callback)
                ),
                VariableInterpolant(
                    descriptor=StrokeMobject._width_,
                    method=VariableInterpolant._wrap_method(SpaceUtils.lerp_callback)
                ),
                VariableInterpolant(
                    descriptor=StrokeMobject._dilate_,
                    method=VariableInterpolant._wrap_method(SpaceUtils.lerp_callback)
                )
            ]
        }
        return tuple(it.chain.from_iterable(
            interpolants
            for parent_cls, interpolants in class_specialized_interpolants_dict.items()
            if issubclass(mobject_cls, parent_cls)
        ))
