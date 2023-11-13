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
    def standardize_color(
        cls: type[Self],
        color: ColorType
    ) -> NP_3f8:
        match color:
            case Color():
                return np.array(color.rgb)
            case str() if re.fullmatch(r"\w+", color):
                return np.array(Color(color).rgb)
            case str() if re.fullmatch(r"#[0-9A-F]+", color, flags=re.IGNORECASE) and (hex_len := len(color) - 1) in (3, 6):
                component_size = hex_len // 3
                return (1.0 / (16 ** component_size - 1)) * np.fromiter((
                    int(match_obj.group(), 16)
                    for match_obj in re.finditer(rf"[0-9A-F]{{{component_size}}}", color, flags=re.IGNORECASE)
                ), dtype=np.float64)
            case np.ndarray() if color.ndim == 1 and color.size == 3:
                return color.astype(np.float64)
            case _:
                raise ValueError(f"Invalid color: {color}")

    @classmethod
    def color_to_hex(
        cls: type[Self],
        color: ColorType
    ) -> str:
        components = (cls.standardize_color(color) * 255.0).astype(np.int32)
        return f"#{"".join("{:02x}".format(component) for component in components)}"
