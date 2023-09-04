from ....constants.custom_typing import (
    ColorT,
    NP_3f8
)
from ....utils.color_utils import ColorUtils
from .array_attribute import ArrayAttribute


class ColorAttribute(ArrayAttribute[NP_3f8]):
    __slots__ = ()

    @classmethod
    def _convert_input(
        cls,
        color_input: ColorT
    ) -> "ColorAttribute":
        return ColorAttribute(ColorUtils.standardize_color(color_input))
