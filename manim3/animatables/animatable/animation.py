from __future__ import annotations


import weakref
from typing import (
    Self,
    TypedDict,
    Unpack
)

from ...constants.custom_typing import RateType
from ...constants.rates import Rates
from ...lazy.lazy_object import LazyObject
from ...timelines.timeline import Timeline


class AnimateKwargs(TypedDict, total=False):
    rate: RateType
    rewind: bool
    infinite: bool


class Animation(LazyObject):
    __slots__ = ()

    def build(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
    ) -> AnimationsTimeline:
        timeline = AnimationsTimeline(**kwargs)
        timeline._animations.append(self)
        return timeline

    def update(
        self: Self,
        alpha: float
    ) -> None:
        pass


class BodyAnimationsTimeline(Timeline):
    __slots__ = ("_animations_timeline_ref",)

    def __init__(
        self: Self,
        animations_timeline: AnimationsTimeline
    ) -> None:
        super().__init__(run_alpha=animations_timeline._run_alpha)
        self._animations_timeline_ref: weakref.ref[AnimationsTimeline] = weakref.ref(animations_timeline)

    def _animation_update(
        self: Self,
        time: float
    ) -> None:
        assert (animations_timeline := self._animations_timeline_ref()) is not None
        animations_timeline.update(time)

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
        rate: RateType = Rates.linear(),
        rewind: bool = False,
        infinite: bool = False
    ) -> None:
        super().__init__(run_alpha=float("inf") if infinite else 1.0)
        if rewind:
            rate = Rates.compose(rate, Rates.rewind())
        self._rate: RateType = rate
        self._animations: list[Animation] = []

    def update(
        self: Self,
        time: float
    ) -> None:
        alpha = self._rate(time)
        for animation in self._animations:
            animation.update(alpha)

    async def construct(
        self: Self
    ) -> None:
        self.update(0.0)
        await self.play(BodyAnimationsTimeline(self))
        self.update(1.0)
