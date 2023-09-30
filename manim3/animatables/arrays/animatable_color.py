from typing import TypeVar

import numpy as np

from ...constants.custom_typing import (
    ColorT,
    NP_3f8
)
from ...lazy.lazy import Lazy
from ...utils.color_utils import ColorUtils
from .animatable_array import AnimatableArray


_AnimatableColorT = TypeVar("_AnimatableColorT", bound="AnimatableColor")


class AnimatableColor(AnimatableArray[NP_3f8]):
    __slots__ = ()

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _array_() -> NP_3f8:
        return np.ones((3,))

    @classmethod
    def _convert_input(
        cls: type[_AnimatableColorT],
        color_input: ColorT
    ) -> _AnimatableColorT:
        return super()._convert_input(ColorUtils.standardize_color(color_input))
