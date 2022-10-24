from __future__ import annotations

from colour import Color

from lights.directional_light_shadow import DirectionalLightShadow
from lights.light import Light
from utils.arrays import Mat4


class DirectionalLight(Light):
    def __init__(self, color: Color, intensity: float = 1.0):
        super().__init__(color, intensity)
        self.cast_shadow: bool = True
        self.target_matrix: Mat4 = Mat4()
        self.shadow: DirectionalLightShadow = DirectionalLightShadow()
