from __future__ import annotations


import re
from typing import Self

import numpy as np
from colour import Color

from ...constants.custom_typing import (
    ColorType,
    NP_3f8
)
from .animatable_array import AnimatableArray


class AnimatableColor(AnimatableArray[NP_3f8]):
    __slots__ = ()

    def __init__(
        self: Self,
        color: ColorType
    ) -> None:
        super().__init__(type(self)._color_to_array(color))

    @classmethod
    def _color_to_array(
        cls: type[Self],
        color: ColorType
    ) -> NP_3f8:
        match color:
            case Color():
                return np.array(color.rgb)
            case str() if re.fullmatch(r"\w+", color):
                return np.array(Color(color).rgb)
            case str() if re.fullmatch(r"#[0-9A-F]{6}", color, flags=re.IGNORECASE):
                return np.fromiter((int(color[start:start + 2], 16) for start in range(1, 7, 2)), dtype=np.float64) / 255.0
            case np.ndarray() if color.shape == (3,):
                return color.astype(np.float64)
            case _:
                raise ValueError(f"Invalid color: {color}")
