from __future__ import annotations


from typing import (
    Iterator,
    Self,
    Unpack
)

import numpy as np

from ...lazy.lazy import Lazy
from ..animatable.actions import Action
from ..animatable.animatable import (
    Animatable,
    AnimatableActions,
    AnimatableInterpolateAnimation,
    DynamicAnimatable
)
from ..animatable.animation import (
    AnimateKwargs,
    Animation
)


class AnimatableArrayActions(AnimatableActions):
    __slots__ = ()

    @Action.register()
    @classmethod
    def interpolate(
        cls: type[Self],
        dst: AnimatableArray,
        src_0: AnimatableArray,
        src_1: AnimatableArray
    ) -> Iterator[Animation]:
        yield AnimatableArrayInterpolateAnimation(dst, src_0, src_1)


class AnimatableArray[NDArrayT: np.ndarray](AnimatableArrayActions, Animatable):
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
    ) -> DynamicAnimatableArray[Self]:
        return DynamicAnimatableArray(self, **kwargs)


class DynamicAnimatableArray[AnimatableArrayT: AnimatableArray](AnimatableArrayActions, DynamicAnimatable[AnimatableArrayT]):
    __slots__ = ()


class AnimatableArrayInterpolateAnimation[AnimatableArrayT: AnimatableArray](AnimatableInterpolateAnimation[AnimatableArrayT]):
    __slots__ = ()

    def interpolate(
        self: Self,
        dst: AnimatableArrayT,
        alpha: float
    ) -> None:
        dst._array_ = (1.0 - alpha) * self._src_0_._array_ + alpha * self._src_1_._array_
