__all__ = ["ColorUtils"]


from abc import (
    ABC,
    abstractmethod
)
import re

from colour import Color
import numpy as np

from ..custom_typing import (
    ColorType,
    Vec3T
)


class ColorUtils(ABC):
    __slots__ = ()

    @abstractmethod
    def __new__(cls) -> None:
        pass

    @classmethod
    def decompose_color(
        cls,
        color: ColorType
    ) -> tuple[Vec3T, float | None]:
        error_message = f"Invalid color: {color}"
        if isinstance(color, Color):
            return np.array(color.rgb), None
        if isinstance(color, str):
            if re.fullmatch(r"\w+", color):
                return np.array(Color(color).rgb), None
            assert re.fullmatch(r"#[0-9A-F]+", color, flags=re.IGNORECASE), error_message
            hex_len = len(color) - 1
            assert hex_len in (3, 4, 6, 8), error_message
            num_components = 4 if hex_len % 3 else 3
            component_size = hex_len // num_components
            result = np.array([
                int(match_obj.group(), 16)
                for match_obj in re.finditer(rf"[0-9A-F]{{{component_size}}}", color, flags=re.IGNORECASE)
            ]) * (1.0 / (16 ** component_size - 1))
            if num_components == 3:
                return result[:], None
            return result[:3], result[3]
        if isinstance(color, np.ndarray):
            if color.shape == (3,):
                return color[:], None
            if color.shape == (4,):
                return color[:3], color[3]
        raise TypeError(error_message)

    @classmethod
    def color_to_hex(
        cls,
        color: ColorType
    ) -> str:
        rgb, _ = cls.decompose_color(color)
        return "#{:02x}{:02x}{:02x}".format(*(rgb * 255.0).astype(int))

    @classmethod
    def normalize_color_input(
        cls,
        color: ColorType | None,
        opacity: float | None
    ) -> tuple[Vec3T | None, float | None]:
        color_component = None
        opacity_component = None
        if color is not None:
            color_component, opacity_component = cls.decompose_color(color)
        if opacity is not None:
            opacity_component = opacity
        return color_component, opacity_component
