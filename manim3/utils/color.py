import re

from colour import Color
import numpy as np

from ..custom_typing import (
    ColorT,
    NP_3f8
)


class ColorUtils:
    __slots__ = ()

    def __new__(cls):
        raise TypeError

    @classmethod
    def standardize_color(
        cls,
        color: ColorT
    ) -> NP_3f8:
        match color:
            case Color():
                return np.array(color.rgb)
            case str() if re.fullmatch(r"\w+", color):
                return np.array(Color(color).rgb)
            case str() if re.fullmatch(r"#[0-9A-F]+", color, flags=re.IGNORECASE) and (hex_len := len(color) - 1) in (3, 6):
                component_size = hex_len // 3
                return (1.0 / (16 ** component_size - 1)) * np.array([
                    int(match_obj.group(), 16)
                    for match_obj in re.finditer(rf"[0-9A-F]{{{component_size}}}", color, flags=re.IGNORECASE)
                ])
            case np.ndarray() if color.ndim == 1 and color.size == 3:
                return color.astype(np.float64)
            case _:
                raise ValueError(f"Invalid color: {color}")

    @classmethod
    def color_to_hex(
        cls,
        color: ColorT
    ) -> str:
        components = (cls.standardize_color(color) * 255.0).astype(int)
        return "#" + "".join("{:02x}".format(component) for component in components)
