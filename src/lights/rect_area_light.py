from __future__ import annotations

from colour import Color

from lights.light import Light


class RectAreaLight(Light):
    def __init__(
        self,
        color: Color,
        intensity: float = 1.0,
        width: float = 10.0,
        height: float = 10.0
    ):
        super().__init__(color, intensity)
        self.width: float = width
        self.height: float = height
