from __future__ import annotations

from colour import Color

from lights.point_light_shadow import PointLightShadow
from lights.light import Light


class PointLight(Light):
    def __init__(
        self,
        color: Color,
        intensity: float = 1.0,
        distance: float = 0.0,
        decay: float = 1.0  # For physically correct lights, should be 2
    ):
        super().__init__(color, intensity)
        self.distance: float = distance
        self.decay: float = decay
        self.cast_shadow: bool = True
        self.shadow: PointLightShadow = PointLightShadow()
