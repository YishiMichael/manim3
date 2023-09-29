from typing import (
    Generic,
    TypeVar
)

import numpy as np

from ...lazy.lazy import Lazy
from ...constants.custom_typing import NP_xf8
from ...utils.space_utils import SpaceUtils
from ..animatable import Updater
from ..leaf_animatable import LeafAnimatable
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
        value: float | np.ndarray | None = None
    ) -> None:
        super().__init__()
        if value is not None:
            self._array_ = np.asarray(value, dtype=np.float64)

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _array_() -> _NPT:
        return NotImplemented

    @classmethod
    def _convert_input(
        cls: type[_AnimatableArrayT],
        array_input: float | np.ndarray
    ) -> _AnimatableArrayT:
        return cls(array_input)

    @classmethod
    def _interpolate(
        cls,
        dst: _AnimatableArrayT,
        src_0: _AnimatableArrayT,
        src_1: _AnimatableArrayT
    ) -> Updater:
        return AnimatableArrayInterpolateUpdater(dst, src_0, src_1)

    @classmethod
    def _split(
        cls,
        dst_tuple: tuple[_AnimatableArrayT, ...],
        src: _AnimatableArrayT,
        alphas: NP_xf8
    ) -> None:
        assert len(dst_tuple) == len(alphas) + 1
        for dst in dst_tuple:
            dst._array_ = src._array_

    @classmethod
    def _concatenate(
        cls,
        dst: _AnimatableArrayT,
        src_tuple: tuple[_AnimatableArrayT, ...]
    ) -> None:
        unique_arrays = {id(src._array_): src._array_ for src in src_tuple}
        if not unique_arrays:
            return
        _, dst._array_ = unique_arrays.popitem()
        assert not unique_arrays


class AnimatableArrayInterpolateUpdater(Updater, Generic[_NPT]):
    __slots__ = ("_animatable_array",)

    def __init__(
        self,
        animatable_array: AnimatableArray[_NPT],
        animatable_array_0: AnimatableArray[_NPT],
        animatable_array_1: AnimatableArray[_NPT]
    ) -> None:
        super().__init__()
        self._animatable_array: AnimatableArray[_NPT] = animatable_array
        self._array_0_ = animatable_array_0._array_
        self._array_1_ = animatable_array_1._array_

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _array_0_() -> _NPT:
        return NotImplemented

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _array_1_() -> _NPT:
        return NotImplemented

    def update(
        self,
        alpha: float
    ) -> None:
        super().update(alpha)
        self._animatable_array._array_ = SpaceUtils.lerp(self._array_0_, self._array_1_, alpha)
        #return AnimatableArray(SpaceUtils.lerp(self._array_0, self._array_1, alpha))

    def initial_update(self) -> None:
        super().initial_update()
        self.update(0.0)

    def final_update(self) -> None:
        super().final_update()
        self.update(1.0)
