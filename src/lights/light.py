from __future__ import annotations

from colour import Color

from mobject import Mobject


class Light(Mobject):
    def __init__(self, color: Color, intensity: float = 1.0):
        self.color: Color = color
        self.intensity: float = intensity
        #self.cast_shadow: bool = True
