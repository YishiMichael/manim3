from __future__ import annotations

from utils.arrays import Vec3


class SphericalHarmonics3:
    def __init__(self):
        self.coefficients: list[Vec3] = [Vec3() for _ in range(9)]
