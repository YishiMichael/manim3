from ...constants.custom_typing import (
    ColorT,
    NP_3f8
)
from ...utils.color_utils import ColorUtils
from .animatable_array import AnimatableArray


class AnimatableColor(AnimatableArray[NP_3f8]):
    __slots__ = ()

    @classmethod
    def _convert_input(
        cls,
        color_input: ColorT
    ) -> "AnimatableColor":
        return AnimatableColor(ColorUtils.standardize_color(color_input))
