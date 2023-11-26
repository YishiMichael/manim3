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
    ) -> DynamicAnimatableArray[Self]:
        return DynamicAnimatableArray(self, **kwargs)

    interpolate = AnimatableArrayActions.interpolate.build_action_descriptor()


class DynamicAnimatableArray[AnimatableArrayT: AnimatableArray](DynamicAnimatable[AnimatableArrayT]):
    __slots__ = ()

    interpolate = AnimatableArrayActions.interpolate.build_dynamic_action_descriptor()


class AnimatableArrayInterpolateAnimation[AnimatableArrayT: AnimatableArray](AnimatableInterpolateAnimation[AnimatableArrayT]):
    __slots__ = ()

    def __init__(
        self: Self,
        dst: AnimatableArrayT,
        src_0: AnimatableArrayT,
        src_1: AnimatableArrayT
    ) -> None:
        super().__init__(dst, src_0, src_1)
        self._array_0_ = src_0._array_
        self._array_1_ = src_1._array_

    @Lazy.variable()
    @staticmethod
    def _array_0_() -> np.ndarray:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _array_1_() -> np.ndarray:
        return NotImplemented

    def interpolate(
        self: Self,
        dst: AnimatableArrayT,
        alpha: float
    ) -> None:
        dst._array_ = (1.0 - alpha) * self._array_0_ + alpha * self._array_1_

    def becomes(
        self: Self,
        dst: AnimatableArrayT,
        src: AnimatableArrayT
    ) -> None:
        dst._array_ = src._array_
