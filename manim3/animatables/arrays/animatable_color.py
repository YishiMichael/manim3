from __future__ import annotations


from typing import Self

from ...constants.custom_typing import (
    ColorType,
    NP_3f8
)
from ...utils.color_utils import ColorUtils
from .animatable_array import AnimatableArray


class AnimatableColor(AnimatableArray[NP_3f8]):
    __slots__ = ()

    def __init__(
        self: Self,
        color: ColorType
    ) -> None:
        super().__init__(ColorUtils.standardize_color(color))
