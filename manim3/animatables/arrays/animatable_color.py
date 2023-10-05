from __future__ import annotations


from typing import Self

import numpy as np

from ...constants.custom_typing import (
    ColorT,
    NP_3f8
)
from ...lazy.lazy import Lazy
from ...utils.color_utils import ColorUtils
from .animatable_array import AnimatableArray


class AnimatableColor(AnimatableArray[NP_3f8]):
    __slots__ = ()

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _array_() -> NP_3f8:
        return np.ones((3,))

    @classmethod
    def _convert_input(
        cls: type[Self],
        color_input: ColorT
    ) -> Self:
        return super()._convert_input(ColorUtils.standardize_color(color_input))
