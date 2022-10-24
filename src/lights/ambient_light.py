from __future__ import annotations

from colour import Color

from lights.light import Light


class AmbientLight(Light):
    def __init__(self, color: Color, intensity: float = 1.0):
        super().__init__(color, intensity)
