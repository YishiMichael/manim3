from __future__ import annotations


import weakref
from abc import ABC
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterable,
    Iterator,
    Self,
    TypedDict,
    Unpack
)

from ...timelines.timeline.rate import Rate
from ...timelines.timeline.rates import Rates
from ...timelines.timeline.timeline import Timeline
#from ..timelines.timeline.rates.linear import Linear
#from ..timelines.timeline.rates.rate import Rate
from ...constants.custom_typing import (
    BoundaryT,
    ColorT
)
from ...lazy.lazy_descriptor import LazyDescriptor
from ...lazy.lazy_object import LazyObject
from .piecewiser import Piecewiser

if TYPE_CHECKING:
    from ..cameras.camera import Camera
    from ..geometries.graph import Graph
    from ..geometries.mesh import Mesh
    from ..geometries.shape import Shape
    from ..lights.ambient_light import AmbientLight
    from ..lights.lighting import Lighting
    from ..lights.point_light import PointLight


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class AnimateInfo:
#    rate: Rate
#    rewind: bool
#    infinite: bool


class AnimateKwargs(TypedDict, total=False):
    rate: Rate
    rewind: bool
    infinite: bool


class SetKwargs(TypedDict, total=False):
    # polymorphism variables
    color: ColorT
    opacity: float
    weight: float

    # Lighting
    ambient_lights: Iterable[AmbientLight]
    point_lights: Iterable[PointLight]

    # Mobject
    camera: Camera

    # MeshMobject
    mesh: Mesh
    ambient_strength: float
    specular_strength: float
    shininess: float
    lighting: Lighting

    # ShapeMobject
    shape: Shape

    # GraphMobject
    graph: Graph
    width: float


class Animatable(LazyObject):
    __slots__ = ()

    _animatable_descriptors: ClassVar[dict[str, LazyDescriptor]] = {}

    #_unanimatable_variable_names: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(
        cls: type[Self]
    ) -> None:
        super().__init_subclass__()
        cls._animatable_descriptors = {
            name: descriptor
            for name, descriptor in cls._lazy_descriptors.items()
            if descriptor._is_variable
            and descriptor._element_type is not None
            and issubclass(descriptor._element_type, Animatable)
            and descriptor._element_type is not cls
        }

    #def __init__(
    #    self: Self
    #) -> None:
    #    super().__init__()
    #    self._animate_info: AnimateInfo | None = None
    #    self._animations: list[Animation] = []
    #    #self._reset_animations()

    #@classmethod
    #def _stack_animations(
    #    cls,
    #    method: "Callable[Concatenate[_T, _P], Iterator[Animation]]"
    #) -> Callable[Concatenate[_T, _P], _T]:

    #    def result(
    #        self: _T,
    #        *args: _P.args,
    #        **kwargs: _P.kwargs
    #    ) -> _T:
    #        for animation in method(self, *args, **kwargs):
    #            self._animations.append(animation)
    #            animation.update(1.0)
    #        return self

    #    return result

    #@classmethod
    #@property
    #def _animatable_descriptors(cls) -> Iterator[LazyDescriptor]:
    #    for descriptor in cls._lazy_descriptors.values():
    #        if descriptor._is_variable and descriptor._element_type is not cls:
    #            yield descriptor

    #def _reset_animations(
    #    self: Self
    #) -> None:
    #    self._animations.clear()

    @classmethod
    def _convert_input(
        cls: type[Self],
        animatable_input: Self
    ) -> Self:
        return animatable_input

    #@classmethod
    #def _iter_elements_from_input(
    #    cls: type[Self],
    #    descriptor: LazyDescriptor,
    #    animatable_input: Any
    #) -> Iterator:
    #    if descriptor._is_multiple:
    #        for animatable_input_element in animatable_input:
    #            yield cls._convert_input(animatable_input_element)
    #    else:
    #        yield cls._convert_input(animatable_input)

    def _iter_siblings(
        self: Self,
        *,
        broadcast: bool = True
    ) -> Iterator[Animatable]:
        yield self

    #@classmethod
    #def _split(
    #    cls: type[_AnimatableT],
    #    dst_tuple: tuple[_AnimatableT, ...],
    #    src: _AnimatableT,
    #    alphas: NP_xf8
    #) -> None:
    #    for descriptor in cls._animatable_descriptors.values():
    #        assert issubclass(element_type := descriptor._element_type, Animatable)
    #        for dst_element_tuple, src_element in zip(
    #            tuple(descriptor._get_elements(dst) for dst in dst_tuple),
    #            descriptor._get_elements(src),
    #            strict=True
    #        ):
    #            element_type._split(dst_element_tuple, src_element, alphas)

    #@classmethod
    #def _concatenate(
    #    cls: type[_AnimatableT],
    #    dst: _AnimatableT,
    #    src_tuple: tuple[_AnimatableT, ...]
    #) -> None:
    #    for descriptor in cls._animatable_descriptors.values():
    #        assert issubclass(element_type := descriptor._element_type, Animatable)
    #        for dst_element, src_element_tuple in zip(
    #            descriptor._get_elements(dst),
    #            tuple(descriptor._get_elements(src) for src in src_tuple),
    #            strict=True
    #        ):
    #            element_type._concatenate(dst_element, src_element_tuple)

    #def _get_interpolate_animation(
    #    self: _AnimatableT,
    #    animatable_0: _AnimatableT,
    #    animatable_1: _AnimatableT
    #) -> Animation:
    #    raise NotImplementedError

    #def interpolate(
    #    self,
    #    animatable_0: _AnimatableT,
    #    animatable_1: _AnimatableT
    #):
    #    self._stack_animation(type(self)._interpolate(
    #        self, animatable_0, animatable_1
    #    ))
    #    return self

    #def submit_timeline(
    #    self: Self
    #) -> Timeline:
    #    #assert not infinite or not rewind
    #    #assert (saved_state := self._saved_state) is not None
    #    #self._copy_lazy_content(self, saved_state)
    #    assert (animate_info := self._animate_info) is not None, "Cannot submit timeline"
    #    rate = animate_info.rate
    #    if animate_info.rewind:
    #        rate = Rates.compose(rate, Rates.rewind())
    #    run_alpha = float("inf") if animate_info.infinite else 1.0
    #    timeline = AnimationsTimeline(
    #        animations=tuple(self._animations),
    #        rate=rate,
    #        run_alpha=run_alpha
    #    )
    #    #timeline = AnimationTimeline(
    #    #    #instance=self,
    #    #    animation=self._animation,
    #    #    rate=rate,
    #    #    run_alpha=float("inf") if infinite else 1.0
    #    #    #infinite=infinite,
    #    #    #run_alpha=float("inf") if infinite else 1.0,
    #    #    #rewind=rewind
    #    #)
    #    self._animate_info = None
    #    #self._animation = Animation()
    #    self._animations.clear()
    #    return timeline

    @property
    def _animate_cls(
        self: Self
    ) -> type[AnimatableAnimationBuilder]:
        return AnimatableAnimationBuilder

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
        #rate: Rate = Rates.linear(),
        #rewind: bool = False,
        #run_alpha: float = 1.0,
        #infinite: bool = False
    ) -> AnimatableAnimationBuilder[Self]:
        return AnimatableAnimationBuilder(self, **kwargs)
        #assert self._animate_info is None, "Existing timeline has not been submitted"
        #self._animate_info = AnimateInfo(
        #    rate=rate,
        #    rewind=rewind,
        #    infinite=infinite
        #)
        #self._animations.clear()
        #return self

    def interpolate(
        self: Self,
        animatable_0: Self,
        animatable_1: Self,
        *,
        broadcast: bool = True,
        alpha: float = 1.0
    ) -> Self:
        self.animate().interpolate(animatable_0, animatable_1, broadcast=broadcast).update(alpha)
        return self

    def piecewise(
        self: Self,
        animatable: Self,
        piecewiser: Piecewiser,
        *,
        broadcast: bool = True,
        alpha: float = 1.0
    ) -> Self:
        self.animate().piecewise(animatable, piecewiser, broadcast=broadcast).update(alpha)
        return self

    def set(
        self: Self,
        *,
        broadcast: bool = True,
        **kwargs: Unpack[SetKwargs]
    ) -> Self:
        #for dst_sibling in self._iter_siblings(broadcast=broadcast):
        #    for name, animatable_input in kwargs.items():
        #        if (descriptor := type(dst_sibling)._animatable_descriptors.get(f"_{name}_")) is None:
        #            continue
        #        assert (element_type := descriptor._element_type) is not None and issubclass(element_type, Animatable)
        #        target_elements = tuple(element_type._iter_elements_from_input(descriptor, animatable_input))
        #        #if descriptor._is_multiple:
        #        #    target_elements = tuple(
        #        #        element_type._convert_input(animatable_input_element)
        #        #        for animatable_input_element in animatable_input
        #        #    )
        #        #else:
        #        #    target_elements = (element_type._convert_input(animatable_input),)
        #        descriptor._set_elements(dst_sibling, target_elements)
        self.animate().set(broadcast=broadcast, **kwargs).update_boundary(1)
        #self._stack_animations(self._iter_set_animations(self._animatable, **kwargs))
        return self

    def transform(
        self: Self,
        animatable: Self,
        *,
        broadcast: bool = True
    ) -> Self:
        self.animate().transform(animatable, broadcast=broadcast).update_boundary(1)
        #self.interpolate(self._animatable.copy(), animatable)
        return self

    #def static_interpolate(
    #    self: Self,
    #    animatable_0: Self,
    #    animatable_1: Self,
    #    alpha: float
    #) -> Self:
    #    for animation in self._iter_interpolate_animations(animatable_0, animatable_1):
    #        animation.update(alpha)
    #    return self

    #def static_piecewise(
    #    self: Self,
    #    animatable: Self,
    #    piecewiser: Piecewiser,
    #    alpha: float
    #) -> Self:
    #    for animation in self._iter_piecewise_animations(animatable, piecewiser):
    #        animation.update(alpha)
    #    return self


class BodyAnimationsTimeline(Timeline):
    __slots__ = ("_animations_timeline_weakref",)

    def __init__(
        self: Self,
        animations_timeline: AnimationsTimeline
        #instance: _T,
        #animations: list[Animation],
        #rate: Rate,
        #run_alpha: float
    ) -> None:
        super().__init__(run_alpha=animations_timeline._run_alpha)
        self._animations_timeline_weakref: weakref.ref[AnimationsTimeline] = weakref.ref(animations_timeline)
        #self._instance: _T = instance
        #self._animations: list[Animation] = animations
        #self._rate: Rate = rate

    def _animation_update(
        self: Self,
        time: float
    ) -> None:
        assert (animations_timeline := self._animations_timeline_weakref()) is not None
        animations_timeline.update(time)
        #self._animation.restore()
        #animations = self._animations
        #animation = self._animation
        #animation.update(animation._rate.at(time))

    async def construct(
        self: Self
    ) -> None:
        await self.wait(self._run_alpha)


class AnimationsTimeline(Timeline):
    __slots__ = (
        "_rate",
        "_animations"
    )

    def __init__(
        self: Self,
        #instance: _T,
        #animation: Animation
        #animation: Animation,
        rate: Rate = Rates.linear(),
        rewind: bool = False,
        #run_alpha: float = 1.0,
        infinite: bool = False
    ) -> None:
        super().__init__(run_alpha=float("inf") if infinite else 1.0)
        #self._instance: _T = instance
        if rewind:
            rate = Rates.compose(rate, Rates.rewind())
        self._rate: Rate = rate
        self._animations: list[Animation] = []
        #self._rate: Rate = rate

    def _stack_animation(
        self: Self,
        animation: Animation
    ) -> None:
        #animation.update_boundary(1)
        self._animations.append(animation)

    def _stack_animations(
        self: Self,
        animations: Iterable[Animation]
    ) -> None:
        self._animations.extend(animations)
        #for animation in animations:
        #    self._stack_animation(animation)
        #    #animation.update_boundary(1)
        #if self._saved_state is not None:
        #self._animations.extend(animations)

    def update(
        self,
        time: float
    ) -> None:
        alpha = self._rate.at(time)
        for animation in self._animations:
            animation.update(alpha)
        #self._animation.update(self._rate.at(alpha))
        #sub_alpha = self._rate.at(alpha)
        #instance = self._instance
        #for animation in self._animations:
        #    animation.update(sub_alpha)

    def update_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> None:
        alpha_boundary = self._rate._boundaries_[boundary]
        for animation in self._animations:
            animation.update_boundary(alpha_boundary)

    async def construct(
        self: Self
    ) -> None:
        #for animation in reversed(self._animations):
        #    animation.initial_update()
        #animations = self._animations
        #animation = self._animation
        #rate = self._rate

        #for animation in reversed(animations):
        #    animation.restore()
        #boundary_0, boundary_1 = rate._boundaries_
        #for animation in animations:
        #    animation.update_boundary(boundary_0)
        self.update_boundary(0)
        #animation.restore()
        #animation.update_boundary(rate.at_boundary(0))
        await self.play(BodyAnimationsTimeline(self))
        #await self.wait(self._run_alpha)
        #animation.restore()

        #for animation in reversed(animations):
        #    animation.restore()
        self.update_boundary(1)
        #for animation in animations:
        #    animation.update_boundary(boundary_1)
        #animation.update_boundary(rate.at_boundary(1))
        #for animation in self._animations:
        #    animation.final_update()


class AnimatableAnimationBuilder[AnimatableT: Animatable](AnimationsTimeline):
    __slots__ = ("_animatable",)

    def __init__(
        self: Self,
        animatable: AnimatableT,
        #instance: _T,
        #animation: Animation
        #animation: Animation,
        **kwargs: Unpack[AnimateKwargs]
        #rate: Rate,
        #rewind: bool,
        #run_alpha: float = 1.0,
        #infinite: bool
    ) -> None:
        super().__init__(**kwargs)
        self._animatable: AnimatableT = animatable

    @classmethod
    def _iter_interpolate_animations(
        cls: type[Self],
        dst: AnimatableT,
        src_0: AnimatableT,
        src_1: AnimatableT
    ) -> Iterator[Animation]:
        for descriptor in type(dst)._animatable_descriptors.values():
            #assert issubclass(descriptor._element_type, Animatable)
            #dst_elements = descriptor._get_elements(dst)
            #src_0_elements = descriptor._get_elements(src_0)
            #src_1_elements = descriptor._get_elements(src_1)
            #if not issubclass(element_type, Animatable) and src_0_elements != src_1_elements:
            #    raise ValueError(f"Uninterpolable variables of `{descriptor._name}` don't match")
            for dst_element, src_0_element, src_1_element in zip(
                descriptor._get_elements(dst),
                descriptor._get_elements(src_0),
                descriptor._get_elements(src_1),
                strict=True
            ):
                yield from dst_element._animate_cls._iter_interpolate_animations(
                    dst=dst_element,
                    src_0=src_0_element,
                    src_1=src_1_element
                )

        #if broadcast:
        #    for dst_sibling, src_0_sibling, src_1_sibling in zip(
        #        dst._iter_siblings(), src_0._iter_siblings(), src_1._iter_siblings()
        #    ):
        #        cls._iter_interpolate_animations(dst_sibling, src_0_sibling, src_1_sibling, broadcast=False)

    @classmethod
    def _iter_piecewise_animations(
        cls: type[Self],
        dst: AnimatableT,
        src: AnimatableT,
        piecewiser: Piecewiser
        #split_alphas: NP_xf8,
        #concatenate_indices: NP_xi4
    ) -> Iterator[Animation]:
        for descriptor in type(dst)._animatable_descriptors.values():
            #assert issubclass(descriptor._element_type, Animatable)
            #dst_elements = descriptor._get_elements(dst)
            #src_elements = descriptor._get_elements(src)
            for dst_element, src_element in zip(
                descriptor._get_elements(dst),
                descriptor._get_elements(src),
                strict=True
            ):
                yield from dst_element._animate_cls._iter_piecewise_animations(
                    dst=dst_element,
                    src=src_element,
                    piecewiser=piecewiser
                )

        #if broadcast:
        #    for dst_sibling, src_sibling in zip(
        #        dst._iter_siblings(), src._iter_siblings()
        #    ):
        #        cls._iter_piecewise_animations(dst_sibling, src_sibling, piecewiser, broadcast=False)
        #pieces = tuple(cls() for _ in range(len(split_alphas) + 1))
        #cls._split(pieces, src, split_alphas)
        #cls._concatenate(dst, tuple(pieces[index] for index in concatenate_indices))

    @classmethod
    def _iter_set_animations(
        cls: type[Self],
        dst: AnimatableT,
        **kwargs: Unpack[SetKwargs]
    ) -> Iterator[Animation]:
        for name, animatable_input in kwargs.items():
            if (descriptor := type(dst)._animatable_descriptors.get(f"_{name}_")) is not None:
                yield AnimatableSetAnimation(dst, descriptor, animatable_input)
            #assert (element_type := descriptor._element_type) is not None and issubclass(element_type, Animatable)
            #source_elements = descriptor._get_elements(dst)
            #target_elements = tuple(element_type._iter_elements_from_input(descriptor, animatable_input))
            ##if descriptor._is_multiple:
            ##    target_elements = tuple(
            ##        element_type._convert_input(animatable_input_element)
            ##        for animatable_input_element in animatable_input
            ##    )
            ##else:
            ##    target_elements = (element_type._convert_input(animatable_input),)
            ##descriptor.__set__(self, target_animatable)
            ##if self._saved_state is not None:
            #for source_element, target_element in zip(source_elements, target_elements, strict=True):
            #    yield from cls._iter_interpolate_animations(
            #        dst=source_element,
            #        src_0=source_element.copy(),
            #        src_1=target_element.copy()
            #    )
        #if broadcast:
        #    for dst_sibling in dst._iter_siblings():
        #        cls._iter_set_animations(dst_sibling, broadcast=False, **kwargs)

    def interpolate(
        self: Self,
        animatable_0: AnimatableT,
        animatable_1: AnimatableT,
        *,
        broadcast: bool = True
    ) -> Self:
        for dst_sibling, src_0_sibling, src_1_sibling in zip(
            self._animatable._iter_siblings(broadcast=broadcast),
            animatable_0._iter_siblings(broadcast=broadcast),
            animatable_1._iter_siblings(broadcast=broadcast)
        ):
            self._stack_animations(dst_sibling._animate_cls._iter_interpolate_animations(
                dst=dst_sibling,
                src_0=src_0_sibling,
                src_1=src_1_sibling
            ))
        return self

    def piecewise(
        self: Self,
        animatable: AnimatableT,
        piecewiser: Piecewiser,
        *,
        broadcast: bool = True
    ) -> Self:
        for dst_sibling, src_sibling in zip(
            self._animatable._iter_siblings(broadcast=broadcast),
            animatable._iter_siblings(broadcast=broadcast)
        ):
            self._stack_animations(dst_sibling._animate_cls._iter_piecewise_animations(
                dst=dst_sibling,
                src=src_sibling,
                piecewiser=piecewiser
            ))
        return self

    def set(
        self: Self,
        *,
        broadcast: bool = True,
        **kwargs: Unpack[SetKwargs]
    ) -> Self:
        for dst_sibling in self._animatable._iter_siblings(broadcast=broadcast):
            self._stack_animations(dst_sibling._animate_cls._iter_set_animations(
                dst=dst_sibling,
                **kwargs
            ))
        return self

    def transform(
        self: Self,
        animatable: AnimatableT,
        *,
        broadcast: bool = True
    ) -> Self:
        self.interpolate(
            animatable_0=self._animatable.copy(),
            animatable_1=animatable,
            broadcast=broadcast
        )
        return self


#class Animation(LazyObject):
#    __slots__ = ()

#    #def __init__(
#    #    self,
#    #    instance: _AnimatableT
#    #) -> None:
#    #    super().__init__()
#    #    self._instance: _AnimatableT = instance

#    @abstractmethod
#    def update(
#        self,
#        alpha: float
#    ) -> None:
#        pass

#    @abstractmethod
#    def initial_update(self) -> None:
#        pass

#    @abstractmethod
#    def final_update(self) -> None:
#        pass


class Animation(ABC):
    __slots__ = ()

    def build(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
        #rate: Rate = Rates.linear(),
        #rewind: bool = False,
        #run_alpha: float = 1.0,
        #infinite: bool = False
    ) -> AnimationsTimeline:
        timeline = AnimationsTimeline(**kwargs)
        timeline._stack_animation(self)
        return timeline

    #def __init__(
    #    self: Self,
    #    rate: Rate = Rates.linear(),
    #    rewind: bool = False,
    #    #run_alpha: float = 1.0,
    #    infinite: bool = False
    #) -> None:
    #    super().__init__()
    #    if rewind:
    #        rate = Rates.compose(rate, Rates.rewind())
    #    run_alpha = float("inf") if infinite else 1.0
    #    self._rate: Rate = rate
    #    self._run_alpha: float = run_alpha

    #def __init__(
    #    self: Self
    #) -> None:
    #    super().__init__()
    #    self._branch_animations: list[Animation] = []

    #def add(
    #    self: Self,
    #    *animations: Animation
    #) -> Self:
    #    self._branch_animations.extend(animations)
    #    return self

    #def clear(
    #    self: Self
    #) -> Self:
    #    self._branch_animations.clear()
    #    return self

    #def build_timeline(
    #    self: Self,
    #    rate: Rate = Rates.linear(),
    #    infinite: bool = False
    #) -> AnimationTimeline:
    #    return AnimationTimeline(
    #        animation=self,
    #        rate=rate,
    #        run_alpha=float("inf") if infinite else 1.0
    #    )

    def update(
        self: Self,
        alpha: float
    ) -> None:
        pass
        #for animation in self._branch_animations:
        #    animation.update(alpha)

    def update_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> None:
        pass
        #for animation in self._branch_animations:
        #    animation.update_boundary(boundary)

    #def restore(
    #    self: Self
    #) -> None:
    #    pass
    #    #for animation in reversed(self._branch_animations):
    #    #    animation.restore()


class AnimatableSetAnimation(Animation):
    __slots__ = (
        "_animatable",
        "_descriptor",
        "_source_elements",
        "_target_elements",
        "_animations"
    )

    def __init__(
        self: Self,
        animatable: Animatable,
        descriptor: LazyDescriptor,
        animatable_input: Any
    ) -> None:

        def iter_elements_from_input(
            animatable_cls: type[Animatable],
            animatable_input: Any,
            is_multiple: bool
        ) -> Iterator:
            if is_multiple:
                for animatable_input_element in animatable_input:
                    yield animatable_cls._convert_input(animatable_input_element)
            else:
                yield animatable_cls._convert_input(animatable_input)

        super().__init__()
        assert (element_type := descriptor._element_type) is not None and issubclass(element_type, Animatable)
        source_elements = descriptor._get_elements(animatable)
        target_elements = tuple(iter_elements_from_input(
            animatable_cls=element_type,
            animatable_input=animatable_input,
            is_multiple=descriptor._is_multiple
        ))
        self._animatable: Animatable = animatable
        self._descriptor: LazyDescriptor = descriptor
        self._source_elements: tuple = source_elements
        self._target_elements: tuple = target_elements
        self._animations: tuple[Animation, ...] | None = None
        #self._positions_0_ = positions_0
        #self._positions_1_ = positions_1
        #self._edges_ = edges

    def update(
        self: Self,
        alpha: float
    ) -> None:
        super().update(alpha)
        if (animations := self._animations) is None:
            animations = tuple(
                animation
                for source_element, target_element in zip(self._source_elements, self._target_elements, strict=True)
                for animation in source_element._animate_cls._iter_interpolate_animations(
                    dst=source_element,
                    src_0=source_element.copy(),
                    src_1=target_element.copy()
                )
            )
            self._animations = animations
        for animation in animations:
            animation.update(alpha)
            #interpolate_info.interpolate(self._dst, alpha)

    def update_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> None:
        super().update_boundary(boundary)
        self._descriptor._set_elements(self._animatable, self._target_elements if boundary else self._source_elements)
