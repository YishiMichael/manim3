from __future__ import annotations


#from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    Iterator,
    Self
)

from ...timelines.timeline.rate import Rate
from ...timelines.timeline.rates import Rates
from ...timelines.timeline.timeline import Timeline
#from ..timelines.timeline.rates.linear import Linear
#from ..timelines.timeline.rates.rate import Rate
from ...constants.custom_typing import BoundaryT
from ...lazy.lazy_descriptor import LazyDescriptor
from ...lazy.lazy_object import LazyObject
from .piecewiser import Piecewiser


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class AnimateInfo:
    rate: Rate
    rewind: bool
    infinite: bool


class Animatable(LazyObject):
    __slots__ = (
        "_animate_info",
        "_animations"
    )

    _special_slot_copiers: ClassVar[dict[str, Callable]] = {
        "_animate_info": lambda o: None,
        "_animations": lambda o: []
    }

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

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._animate_info: AnimateInfo | None = None
        self._animations: list[Animation] = []
        #self._reset_animations()

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

    def _stack_animation(
        self: Self,
        animation: Animation
    ) -> None:
        animation.update_boundary(1)
        self._animations.append(animation)

    def _stack_animations(
        self: Self,
        animations: Iterable[Animation]
    ) -> None:
        for animation in animations:
            self._stack_animation(animation)
            #animation.update_boundary(1)
        #if self._saved_state is not None:
        #self._animations.extend(animations)

    def _set_animate_info(
        self: Self,
        animate_info: AnimateInfo
    ) -> None:
        assert self._animate_info is None, "Existing timeline has not been submitted"
        self._animate_info = animate_info
        self._animations.clear()

    def _submit_timeline(
        self: Self
    ) -> Timeline:
        #assert not infinite or not rewind
        #assert (saved_state := self._saved_state) is not None
        #self._copy_lazy_content(self, saved_state)
        assert (animate_info := self._animate_info) is not None, "Cannot submit simeline"
        rate = animate_info.rate
        if animate_info.rewind:
            rate = Rates.compose(rate, Rates.rewind())
        run_alpha = float("inf") if animate_info.infinite else 1.0
        timeline = AnimationsTimeline(
            animations=tuple(self._animations),
            rate=rate,
            run_alpha=run_alpha
        )
        #timeline = AnimationTimeline(
        #    #instance=self,
        #    animation=self._animation,
        #    rate=rate,
        #    run_alpha=float("inf") if infinite else 1.0
        #    #infinite=infinite,
        #    #run_alpha=float("inf") if infinite else 1.0,
        #    #rewind=rewind
        #)
        self._animate_info = None
        #self._animation = Animation()
        self._animations.clear()
        return timeline

    @classmethod
    def _convert_input(
        cls: type[Self],
        animatable_input: Self
    ) -> Self:
        return animatable_input

    def _iter_interpolate_animations(
        self: Self,
        src_0: Self,
        src_1: Self
    ) -> Iterator[Animation]:
        for descriptor in type(self)._animatable_descriptors.values():
            #assert issubclass(descriptor._element_type, Animatable)
            dst_elements = descriptor._get_elements(self)
            src_0_elements = descriptor._get_elements(src_0)
            src_1_elements = descriptor._get_elements(src_1)
            #if not issubclass(element_type, Animatable) and src_0_elements != src_1_elements:
            #    raise ValueError(f"Uninterpolable variables of `{descriptor._name}` don't match")
            for dst_element, src_0_element, src_1_element in zip(
                dst_elements, src_0_elements, src_1_elements, strict=True
            ):
                yield from dst_element._iter_interpolate_animations(src_0_element, src_1_element)

    def _iter_piecewise_animations(
        self: Self,
        src: Self,
        piecewiser: Piecewiser
        #split_alphas: NP_xf8,
        #concatenate_indices: NP_xi4
    ) -> Iterator[Animation]:
        for descriptor in type(self)._animatable_descriptors.values():
            #assert issubclass(descriptor._element_type, Animatable)
            dst_elements = descriptor._get_elements(self)
            src_elements = descriptor._get_elements(src)
            for dst_element, src_element in zip(
                dst_elements, src_elements, strict=True
            ):
                yield from dst_element._iter_piecewise_animations(src_element, piecewiser)
        #pieces = tuple(cls() for _ in range(len(split_alphas) + 1))
        #cls._split(pieces, src, split_alphas)
        #cls._concatenate(dst, tuple(pieces[index] for index in concatenate_indices))

    def _iter_set_animations(
        self: Self,
        **kwargs: Any
    ) -> Iterator[Animation]:
        for attribute_name, animatable_input in kwargs.items():
            if (descriptor := type(self)._animatable_descriptors.get(f"_{attribute_name}_")) is None:
                continue
            assert (element_type := descriptor._element_type) is not None and issubclass(element_type, Animatable)
            source_elements = descriptor._get_elements(self)
            if descriptor._is_multiple:
                target_elements = tuple(
                    element_type._convert_input(animatable_input_element)
                    for animatable_input_element in animatable_input
                )
            else:
                target_elements = (element_type._convert_input(animatable_input),)
            #descriptor.__set__(self, target_animatable)
            #if self._saved_state is not None:
            for source_element, target_element in zip(source_elements, target_elements, strict=True):
                yield from source_element._iter_interpolate_animations(source_element.copy(), target_element.copy())

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

    def animate(
        self: Self,
        rate: Rate = Rates.linear(),
        rewind: bool = False,
        #run_alpha: float = 1.0,
        infinite: bool = False
    ) -> Self:
        self._set_animate_info(AnimateInfo(
            rate=rate,
            rewind=rewind,
            infinite=infinite
        ))
        return self

    def interpolate(
        self: Self,
        animatable_0: Self,
        animatable_1: Self
    ) -> Self:
        self._stack_animations(self._iter_interpolate_animations(animatable_0, animatable_1))
        return self

    def piecewise(
        self: Self,
        animatable: Self,
        piecewiser: Piecewiser
    ) -> Self:
        self._stack_animations(self._iter_piecewise_animations(animatable, piecewiser))
        return self

    def set(
        self: Self,
        **kwargs: Any
    ) -> Self:
        self._stack_animations(self._iter_set_animations(**kwargs))
        return self

    def transform(
        self: Self,
        animatable: Self
    ) -> Self:
        self.interpolate(self.copy(), animatable)
        return self

    def static_interpolate(
        self: Self,
        animatable_0: Self,
        animatable_1: Self,
        alpha: float
    ) -> Self:
        for animation in self._iter_interpolate_animations(animatable_0, animatable_1):
            animation.update(alpha)
        return self

    def static_piecewise(
        self: Self,
        animatable: Self,
        piecewiser: Piecewiser,
        alpha: float
    ) -> Self:
        for animation in self._iter_piecewise_animations(animatable, piecewiser):
            animation.update(alpha)
        return self


class LeafAnimationsTimeline(Timeline):
    __slots__ = (
        #"_instance",
        "_animations",
        "_rate"
    )

    def __init__(
        self: Self,
        #instance: _T,
        animations: tuple[Animation, ...],
        #animation: Animation,
        rate: Rate,
        run_alpha: float
    ) -> None:
        super().__init__(run_alpha=run_alpha)
        #self._instance: _T = instance
        self._animations: tuple[Animation, ...] = animations
        self._rate: Rate = rate

    def _update(
        self: Self,
        alpha: float
    ) -> None:
        #self._animation.restore()
        animations = self._animations
        rated_alpha = self._rate.at(alpha)

        for animation in reversed(animations):
            animation.restore()
        for animation in animations:
            animation.update(rated_alpha)

    async def construct(
        self: Self
    ) -> None:
        await self.wait(self._run_alpha)


class AnimationsTimeline(Timeline):
    __slots__ = (
        #"_instance",
        "_animations",
        "_rate"
    )

    def __init__(
        self: Self,
        #instance: _T,
        animations: tuple[Animation, ...],
        #animation: Animation,
        rate: Rate,
        run_alpha: float
    ) -> None:
        super().__init__(run_alpha=run_alpha)
        #self._instance: _T = instance
        self._animations: tuple[Animation, ...] = animations
        self._rate: Rate = rate

    #def update(
    #    self,
    #    alpha: float
    #) -> None:
    #    self._animation.update(self._rate.at(alpha))
    #    #sub_alpha = self._rate.at(alpha)
    #    #instance = self._instance
    #    #for animation in self._animations:
    #    #    animation.update(sub_alpha)

    async def construct(
        self: Self
    ) -> None:
        #for animation in reversed(self._animations):
        #    animation.initial_update()
        animations = self._animations
        rate = self._rate

        for animation in reversed(animations):
            animation.restore()
        for animation in animations:
            animation.update_boundary(rate.at_boundary(0))
        #animation.restore()
        #animation.update_boundary(rate.at_boundary(0))
        await self.play(LeafAnimationsTimeline(animations, rate, self._run_alpha))
        #await self.wait(self._run_alpha)
        #animation.restore()

        for animation in reversed(animations):
            animation.restore()
        for animation in animations:
            animation.update_boundary(rate.at_boundary(1))
        #animation.update_boundary(rate.at_boundary(1))
        #for animation in self._animations:
        #    animation.final_update()


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


class Animation(LazyObject):
    __slots__ = ()

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

    def restore(
        self: Self
    ) -> None:
        pass
        #for animation in reversed(self._branch_animations):
        #    animation.restore()
