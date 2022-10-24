from __future__ import annotations

from colour import Color

from constants import PI
from lights.spot_light_shadow import SpotLightShadow
from lights.light import Light
from utils.texture import Texture
from utils.arrays import Mat4


class SpotLight(Light):
    def __init__(
        self,
        color: Color,
        intensity: float = 1.0,
        distance: float = 0.0,
        angle: float = PI / 3,
        penumbra: float = 0.0,
        decay: float = 1.0  # For physically correct lights, should be 2
    ):
        super().__init__(color, intensity)
        self.distance: float = distance
        self.angle: float = angle
        self.penumbra: float = penumbra
        self.decay: float = decay
        self.map: Texture | None = None
        self.cast_shadow: bool = True
        self.target_matrix: Mat4 = Mat4()
        self.shadow: SpotLightShadow = SpotLightShadow()
