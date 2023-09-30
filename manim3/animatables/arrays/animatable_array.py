from typing import (
    Generic,
    TypeVar
)

import numpy as np

from ...constants.custom_typing import NP_xf8
from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
from ..animatable.leaf_animatable import (
    LeafAnimatable,
    LeafAnimatableInterpolateInfo
)
#from .mobject_attribute import (
#    InterpolateHandler,
#    MobjectAttribute
#)


_NPT = TypeVar("_NPT", bound=np.ndarray)
_AnimatableArrayT = TypeVar("_AnimatableArrayT", bound="AnimatableArray")


class AnimatableArray(LeafAnimatable, Generic[_NPT]):
    __slots__ = ()

    def __init__(
        self,
        array: _NPT | None = None
    ) -> None:
        super().__init__()
        if array is not None:
            self._array_ = array

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _array_() -> _NPT:
        return NotImplemented

    @classmethod
    def _convert_input(
        cls: type[_AnimatableArrayT],
        array_input: np.ndarray
    ) -> _AnimatableArrayT:
        return cls(array_input)

    @classmethod
    def _interpolate(
        cls: type[_AnimatableArrayT],
        src_0: _AnimatableArrayT,
        src_1: _AnimatableArrayT
    ) -> "AnimatableArrayInterpolateInfo[_AnimatableArrayT]":
        return AnimatableArrayInterpolateInfo(src_0, src_1)

    #@classmethod
    #def _interpolate(
    #    cls: type[_AnimatableArrayT],
    #    dst: _AnimatableArrayT,
    #    src_0: _AnimatableArrayT,
    #    src_1: _AnimatableArrayT
    #) -> Updater:
    #    return AnimatableArrayInterpolateUpdater(dst, src_0, src_1)

    @classmethod
    def _split(
        cls: type[_AnimatableArrayT],
        #dst_tuple: tuple[_AnimatableArrayT, ...],
        src: _AnimatableArrayT,
        alphas: NP_xf8
    ) -> tuple[_AnimatableArrayT, ...]:
        return tuple(cls(src._array_) for _ in range(len(alphas) + 1))
        #assert len(dst_tuple) == len(alphas) + 1
        #for dst in dst_tuple:
        #    dst._array_ = src._array_

    @classmethod
    def _concatenate(
        cls: type[_AnimatableArrayT],
        #dst: _AnimatableArrayT,
        src_tuple: tuple[_AnimatableArrayT, ...]
    ) -> _AnimatableArrayT:
        unique_arrays = {id(src._array_): src._array_ for src in src_tuple}
        if not unique_arrays:
            return cls()
        _, unique_array = unique_arrays.popitem()
        assert not unique_arrays
        return cls(unique_array)


class AnimatableArrayInterpolateInfo(LeafAnimatableInterpolateInfo[_AnimatableArrayT]):
    __slots__ = (
        "_array_0",
        "_array_1"
    )

    def __init__(
        self,
        src_0: _AnimatableArrayT,
        src_1: _AnimatableArrayT
    ) -> None:
        super().__init__(src_0, src_1)
        self._array_0: np.ndarray = src_0._array_
        self._array_1: np.ndarray = src_1._array_

    def interpolate(
        self,
        src: _AnimatableArrayT,
        alpha: float
    ) -> None:
        src._array_ = SpaceUtils.lerp(self._array_0, self._array_1, alpha)


#class AnimatableArrayInterpolateUpdater(Updater, Generic[_NPT]):
#    __slots__ = ("_animatable_array",)

#    def __init__(
#        self,
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
#        self,
#        alpha: float
#    ) -> None:
#        super().update(alpha)
#        self._animatable_array._array_ = SpaceUtils.lerp(self._array_0_, self._array_1_, alpha)
#        #return AnimatableArray(SpaceUtils.lerp(self._array_0, self._array_1, alpha))

#    def update_boundary(
#        self,
#        boundary: BoundaryT
#    ) -> None:
#        super().update_boundary(boundary)
#        self.update(float(boundary))
