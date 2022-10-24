from __future__ import annotations

from colour import Color

from lights.light import Light
#from utils.spherical_harmonics3 import SphericalHarmonics3


class HemisphereLightProbe(Light):
    def __init__(self, sky_color: Color, ground_color: Color, intensity: float = 1.0):
        super().__init__(sky_color, intensity)
        self.ground_color: Color = ground_color
    #def __init__(self, sh: SphericalHarmonics3 | None = None, intensity: float = 1.0):
    #    super().__init__()
    #    if sh is None:
    #        sh = SphericalHarmonics3()
    #    self.sh: SphericalHarmonics3 = sh
    #    self.intensity: float = intensity
