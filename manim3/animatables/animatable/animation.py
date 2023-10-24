from __future__ import annotations


import weakref
from typing import (
    Self,
    TypedDict,
    Unpack
)

from ...constants.custom_typing import BoundaryT
from ...lazy.lazy_object import LazyObject
from ...timelines.timeline.rate import Rate
from ...timelines.timeline.rates import Rates
from ...timelines.timeline.timeline import Timeline


class AnimateKwargs(TypedDict, total=False):
    rate: Rate
    rewind: bool
    infinite: bool


class Animation(LazyObject):
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
        timeline._animations.append(self)
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


class BodyAnimationsTimeline(Timeline):
    __slots__ = ("_animations_timeline_ref",)

    def __init__(
        self: Self,
        animations_timeline: AnimationsTimeline
        #instance: _T,
        #animations: list[Animation],
        #rate: Rate,
        #run_alpha: float
    ) -> None:
        super().__init__(run_alpha=animations_timeline._run_alpha)
        self._animations_timeline_ref: weakref.ref[AnimationsTimeline] = weakref.ref(animations_timeline)
        #self._instance: _T = instance
        #self._animations: list[Animation] = animations
        #self._rate: Rate = rate

    def _animation_update(
        self: Self,
        time: float
    ) -> None:
        assert (animations_timeline := self._animations_timeline_ref()) is not None
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
        #"_animatable",
        "_rate",
        "_animations"
    )

    def __init__(
        self: Self,
        #animatable: AnimatableT,
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
        #self._animatable: AnimatableT = animatable
        self._rate: Rate = rate
        self._animations: list[Animation] = []
        #self._rate: Rate = rate

    #def _stack_animation(
    #    self: Self,
    #    animation: Animation
    #) -> None:
    #    #animation.update_boundary(1)
    #    self._animations.append(animation)

    #def _stack_animations(
    #    self: Self,
    #    animations: Iterable[Animation]
    #) -> None:
    #    self._animations.extend(animations)
    #    #for animation in animations:
    #    #    self._stack_animation(animation)
    #    #    #animation.update_boundary(1)
    #    #if self._saved_state is not None:
    #    #self._animations.extend(animations)

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
