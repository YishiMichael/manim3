from __future__ import annotations


from typing import (
    Iterator,
    Self,
    Unpack
)

import numpy as np

from ...lazy.lazy import Lazy
from ..animatable.action import (
    DescriptiveAction,
    DescriptorParameters
)
from ..animatable.animatable import (
    Animatable,
    AnimatableInterpolateAnimation,
    AnimatableTimeline
)
from ..animatable.animation import (
    AnimateKwargs,
    Animation
)


class AnimatableArray[NDArrayT: np.ndarray](Animatable):
    __slots__ = ()

    def __init__(
        self: Self,
        array: NDArrayT | None = None
    ) -> None:
        super().__init__()
        if array is not None:
            self._array_ = array

    @Lazy.variable()
    @staticmethod
    def _array_() -> NDArrayT:
        return NotImplemented

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
    ) -> AnimatableArrayTimeline[Self]:
        return AnimatableArrayTimeline(self, **kwargs)

    @DescriptiveAction.descriptive_register(DescriptorParameters)
    @classmethod
    def interpolate(
        cls: type[Self],
        dst: Self,
        src_0: Self,
        src_1: Self
    ) -> Iterator[Animation]:
        yield AnimatableArrayInterpolateAnimation(dst, src_0, src_1)


class AnimatableArrayTimeline[AnimatableArrayT: AnimatableArray](AnimatableTimeline[AnimatableArrayT]):
    __slots__ = ()

    interpolate = AnimatableArray.interpolate


class AnimatableArrayInterpolateAnimation[AnimatableArrayT: AnimatableArray](AnimatableInterpolateAnimation[AnimatableArrayT]):
    __slots__ = ()

    def interpolate(
        self: Self,
        dst: AnimatableArrayT,
        alpha: float
    ) -> None:
        dst._array_ = (1.0 - alpha) * self._src_0_._array_ + alpha * self._src_1_._array_
