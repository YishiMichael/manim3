from __future__ import annotations

from colour import Color

from lights.light import Light


class HemisphereLight(Light):
    def __init__(self, sky_color: Color, ground_color: Color, intensity: float = 1.0):
        super().__init__(sky_color, intensity)
        self.ground_color: Color = ground_color
