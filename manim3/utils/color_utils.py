from __future__ import annotations


import re
from typing import (
    Never,
    Self
)

import numpy as np
from colour import Color

from ..constants.custom_typing import (
    ColorType,
    NP_3f8
)


class ColorUtils:
    __slots__ = ()

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @classmethod
    def color_to_array(
        cls: type[Self],
        color: ColorType
    ) -> NP_3f8:
        match color:
            case Color():
                return np.array(color.rgb)
            case str() if re.fullmatch(r"\w+", color):
                return np.array(Color(color).rgb)
            case str() if re.fullmatch(r"#[0-9A-F]{6}", color, flags=re.IGNORECASE):
                return np.fromiter((
                    int(match.group(), 16)
                    for match in re.finditer(r"[0-9A-F]{2}", color, flags=re.IGNORECASE)
                ), dtype=np.float64) / 255.0
            case np.ndarray() if color.shape == (3,):
                return color.astype(np.float64)
            case _:
                raise ValueError(f"Invalid color: {color}")

    @classmethod
    def color_to_hex(
        cls: type[Self],
        color: ColorType
    ) -> str:
        components = (cls.color_to_array(color) * 255.0).astype(np.int32)
        return f"#{"".join("{:02x}".format(component) for component in components)}"
