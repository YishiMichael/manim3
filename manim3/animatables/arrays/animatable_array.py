from typing import (
    Generic,
    TypeVar
)

import numpy as np

from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
from ..animatable import (
    Animatable,
    Updater
)
#from .mobject_attribute import (
#    InterpolateHandler,
#    MobjectAttribute
#)


_NPT = TypeVar("_NPT", bound=np.ndarray)


class AnimatableArray(Animatable, Generic[_NPT]):
    __slots__ = ()

    def __init__(
        self,
        value: float | np.ndarray
    ) -> None:
        super().__init__()
        self._array_ = np.asarray(value, dtype=np.float64)

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _array_() -> _NPT:
        return NotImplemented

    @classmethod
    def _convert_input(
        cls,
        array_input: float | np.ndarray
    ) -> "AnimatableArray[_NPT]":
        return AnimatableArray(array_input)

    def _get_interpolate_updater(
        self,
        array_0: "AnimatableArray[_NPT]",
        array_1: "AnimatableArray[_NPT]"
    ) -> "AnimatableArrayInterpolateUpdater":
        return AnimatableArrayInterpolateUpdater(self, array_0, array_1)


class AnimatableArrayInterpolateUpdater(Updater[AnimatableArray[_NPT]]):
    __slots__ = (
        "_array",
        "_array_0",
        "_array_1"
    )

    def __init__(
        self,
        animatable_array: AnimatableArray[_NPT],
        animatable_array_0: AnimatableArray[_NPT],
        animatable_array_1: AnimatableArray[_NPT]
    ) -> None:
        super().__init__(animatable_array)
        self._array_0: _NPT = animatable_array_0._array_
        self._array_1: _NPT = animatable_array_1._array_

    def update(
        self,
        alpha: float
    ) -> None:
        self._instance._array_ = SpaceUtils.lerp(self._array_0, self._array_1, alpha)
        #return AnimatableArray(SpaceUtils.lerp(self._array_0, self._array_1, alpha))
