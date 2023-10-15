from __future__ import annotations


from typing import Self

import numpy as np

from ...constants.custom_typing import NP_xf8
from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
from ..animatable.leaf_animatable import (
    LeafAnimatable,
    LeafAnimatableInterpolateInfo
)


class AnimatableArray[NDArrayT: np.ndarray](LeafAnimatable):
    __slots__ = ()

    def __init__(
        self: Self,
        array: NDArrayT | None = None
    ) -> None:
        super().__init__()
        if array is not None:
            self._array_ = array

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _array_() -> NDArrayT:
        return NotImplemented

    @classmethod
    def _interpolate(
        cls: type[Self],
        src_0: Self,
        src_1: Self
    ) -> AnimatableArrayInterpolateInfo[Self]:
        return AnimatableArrayInterpolateInfo(src_0, src_1)

    #@classmethod
    #def _interpolate(
    #    cls: type[Self],
    #    dst: Self,
    #    src_0: Self,
    #    src_1: Self
    #) -> Updater:
    #    return AnimatableArrayInterpolateUpdater(dst, src_0, src_1)

    @classmethod
    def _split(
        cls: type[Self],
        #dst_tuple: tuple[Self, ...],
        src: Self,
        alphas: NP_xf8
    ) -> tuple[Self, ...]:
        return tuple(cls(src._array_) for _ in range(len(alphas) + 1))
        #assert len(dst_tuple) == len(alphas) + 1
        #for dst in dst_tuple:
        #    dst._array_ = src._array_

    @classmethod
    def _concatenate(
        cls: type[Self],
        #dst: Self,
        src_tuple: tuple[Self, ...]
    ) -> Self:
        unique_arrays = {id(src._array_): src._array_ for src in src_tuple}
        if not unique_arrays:
            return cls()
        _, unique_array = unique_arrays.popitem()
        assert not unique_arrays
        return cls(unique_array)


class AnimatableArrayInterpolateInfo[AnimatableArrayT: AnimatableArray](LeafAnimatableInterpolateInfo[AnimatableArrayT]):
    __slots__ = (
        "_array_0",
        "_array_1"
    )

    def __init__(
        self: Self,
        src_0: AnimatableArrayT,
        src_1: AnimatableArrayT
    ) -> None:
        super().__init__()
        self._array_0: np.ndarray = src_0._array_
        self._array_1: np.ndarray = src_1._array_

    def interpolate(
        self: Self,
        src: AnimatableArrayT,
        alpha: float
    ) -> None:
        src._array_ = SpaceUtils.lerp(self._array_0, self._array_1, alpha)


#class AnimatableArrayInterpolateUpdater(Updater, Generic[_NPT]):
#    __slots__ = ("_animatable_array",)

#    def __init__(
#        self: Self,
#        animatable_array: AnimatableArray[_NPT],
#        animatable_array_0: AnimatableArray[_NPT],
#        animatable_array_1: AnimatableArray[_NPT]
#    ) -> None:
#        super().__init__()
#        self._animatable_array: AnimatableArray[_NPT] = animatable_array
#        self._array_0_ = animatable_array_0._array_
#        self._array_1_ = animatable_array_1._array_

#    @Lazy.variable(hasher=Lazy.array_hasher)
#    @staticmethod
#    def _array_0_() -> _NPT:
#        return NotImplemented

#    @Lazy.variable(hasher=Lazy.array_hasher)
#    @staticmethod
#    def _array_1_() -> _NPT:
#        return NotImplemented

#    def update(
#        self: Self,
#        alpha: float
#    ) -> None:
#        super().update(alpha)
#        self._animatable_array._array_ = SpaceUtils.lerp(self._array_0_, self._array_1_, alpha)
#        #return AnimatableArray(SpaceUtils.lerp(self._array_0, self._array_1, alpha))

#    def update_boundary(
#        self: Self,
#        boundary: BoundaryT
#    ) -> None:
#        super().update_boundary(boundary)
#        self.update(float(boundary))
