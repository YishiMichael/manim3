import re

from colour import Color
import numpy as np

from ..custom_typing import (
    ColorT,
    Vec3T
)


class ColorUtils:
    __slots__ = ()

    def __new__(cls):
        raise TypeError

    @classmethod
    def decompose_color(
        cls,
        color: ColorT
    ) -> tuple[Vec3T, float | None]:
        match color:
            case Color():
                return np.array(color.rgb), None
            case str() if re.fullmatch(r"\w+", color):
                return np.array(Color(color).rgb), None
            case str() if re.fullmatch(r"#[0-9A-F]+", color, flags=re.IGNORECASE) and (hex_len := len(color) - 1) in (3, 4, 6, 8):
                num_components = 4 if hex_len % 3 else 3
                component_size = hex_len // num_components
                result = (1.0 / (16 ** component_size - 1)) * np.array([
                    int(match_obj.group(), 16)
                    for match_obj in re.finditer(rf"[0-9A-F]{{{component_size}}}", color, flags=re.IGNORECASE)
                ])
                if num_components == 3:
                    return result[:], None
                return result[:3], result[3]
            case np.ndarray() if color.ndim == 1 and (size := color.size) in (3, 4):
                if size == 3:
                    return color[:], None
                return color[:3], color[3]
            case _:
                raise ValueError(f"Invalid color: {color}")

    @classmethod
    def color_to_hex(
        cls,
        color: ColorT,
        *,
        include_alpha: bool = False
    ) -> str:
        rgb, alpha = cls.decompose_color(color)
        if alpha is None:
            alpha = 1.0
        if include_alpha:
            components = np.append(rgb, alpha)
        else:
            components = rgb
        components = (components * 255.0).astype(int)
        return "#" + "".join("{:02x}".format(component) for component in components)

    @classmethod
    def standardize_color_input(
        cls,
        color: ColorT | None,
        opacity: float | None
    ) -> tuple[Vec3T | None, float | None]:
        color_component = None
        opacity_component = None
        if color is not None:
            color_component, opacity_component = cls.decompose_color(color)
        if opacity is not None:
            opacity_component = opacity
        return color_component, opacity_component
