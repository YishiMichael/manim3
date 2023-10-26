from __future__ import annotations


from typing import (
    Iterator,
    Self,
    Unpack,
    override
)

#import numpy as np
import numpy.typing as npt

#from ...constants.custom_typing import NP_xf8
from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
from ..animatable.actions import ActionMeta
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
#from ..animatable.leaf_animatable import (
#    LeafAnimatable,
#    LeafAnimatableInterpolateInfo
#)


class AnimatableArrayActions(AnimatableActions):
    __slots__ = ()

    @ActionMeta.register
    @classmethod
    @override
    def interpolate(
        cls: type[Self],
        dst: AnimatableArray,
        src_0: AnimatableArray,
        src_1: AnimatableArray
    ) -> Iterator[Animation]:
        yield AnimatableArrayInterpolateAnimation(dst, src_0, src_1)


class AnimatableArray[NDArrayT: npt.NDArray](AnimatableArrayActions, Animatable):
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
        #rate: Rate = Rates.linear(),
        #rewind: bool = False,
        #run_alpha: float = 1.0,
        #infinite: bool = False
    ) -> DynamicAnimatableArray[Self]:
        return DynamicAnimatableArray(self, **kwargs)

    #@classmethod
    #def _interpolate(
    #    cls: type[Self],
    #    src_0: Self,
    #    src_1: Self
    #) -> ArrayInterpolateInfo[Self]:
    #    return ArrayInterpolateInfo(src_0, src_1)

    #@classmethod
    #def _interpolate(
    #    cls: type[Self],
    #    dst: Self,
    #    src_0: Self,
    #    src_1: Self
    #) -> Updater:
    #    return ArrayInterpolateUpdater(dst, src_0, src_1)

    #@classmethod
    #def _split(
    #    cls: type[Self],
    #    #dst_tuple: tuple[Self, ...],
    #    src: Self,
    #    alphas: NP_xf8
    #) -> tuple[Self, ...]:
    #    return tuple(cls(src._array_) for _ in range(len(alphas) + 1))
    #    #assert len(dst_tuple) == len(alphas) + 1
    #    #for dst in dst_tuple:
    #    #    dst._array_ = src._array_

    #@classmethod
    #def _concatenate(
    #    cls: type[Self],
    #    #dst: Self,
    #    srcs: tuple[Self, ...]
    #) -> Self:
    #    unique_arrays = {id(src._array_): src._array_ for src in srcs}
    #    if not unique_arrays:
    #        return cls()
    #    _, unique_array = unique_arrays.popitem()
    #    assert not unique_arrays
    #    return cls(unique_array)


class DynamicAnimatableArray[AnimatableArrayT: AnimatableArray](AnimatableArrayActions, DynamicAnimatable[AnimatableArrayT]):
    __slots__ = ()


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
    def _array_0_() -> npt.NDArray:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _array_1_() -> npt.NDArray:
        return NotImplemented

    def interpolate(
        self: Self,
        dst: AnimatableArrayT,
        alpha: float
    ) -> None:
        dst._array_ = SpaceUtils.lerp(self._array_0_, self._array_1_, alpha)

    def becomes(
        self: Self,
        dst: AnimatableArrayT,
        src: AnimatableArrayT
    ) -> None:
        dst._array_ = src._array_


#class ArrayInterpolateInfo[ArrayT: Array](LeafAnimatableInterpolateInfo[ArrayT]):
#    __slots__ = (
#        "_array_0",
#        "_array_1"
#    )

#    def __init__(
#        self: Self,
#        src_0: ArrayT,
#        src_1: ArrayT
#    ) -> None:
#        super().__init__()
#        self._array_0: np.ndarray = src_0._array_
#        self._array_1: np.ndarray = src_1._array_

#    def interpolate(
#        self: Self,
#        dst: ArrayT,
#        alpha: float
#    ) -> None:
#        dst._array_ = SpaceUtils.lerp(self._array_0, self._array_1, alpha)


#class ArrayInterpolateUpdater(Updater, Generic[_NPT]):
#    __slots__ = ("_animatable_array",)

#    def __init__(
#        self: Self,
#        animatable_array: Array[_NPT],
#        animatable_array_0: Array[_NPT],
#        animatable_array_1: Array[_NPT]
#    ) -> None:
#        super().__init__()
#        self._animatable_array: Array[_NPT] = animatable_array
#        self._array_0_ = animatable_array_0._array_
#        self._array_1_ = animatable_array_1._array_

#    @Lazy.variable()
#    @staticmethod
#    def _array_0_() -> _NPT:
#        return NotImplemented

#    @Lazy.variable()
#    @staticmethod
#    def _array_1_() -> _NPT:
#        return NotImplemented

#    def update(
#        self: Self,
#        alpha: float
#    ) -> None:
#        super().update(alpha)
#        self._animatable_array._array_ = SpaceUtils.lerp(self._array_0_, self._array_1_, alpha)
#        #return Array(SpaceUtils.lerp(self._array_0, self._array_1, alpha))

#    def update_boundary(
#        self: Self,
#        boundary: BoundaryT
#    ) -> None:
#        super().update_boundary(boundary)
#        self.update(float(boundary))
