from dataclasses import dataclass
from typing import Self

import numpy as np

from cameras.camera import Camera
from utils.arrays import Mat4


@dataclass
class PerspectiveCamera(Camera):
    fov: float = 50.0
    aspect: float = 1.0
    near: float = 0.1
    far: float = 2000.0

    def calculate_projection_matrix(self: Self) -> Mat4:
        DEGREES = np.pi / 180.0
        near = self.near
        top = near * np.tan(0.5 * self.fov * DEGREES) / self.zoom
        height = 2 * top
        width = self.aspect * height
        left = - 0.5 * width + self.filmOffset / self.filmGauge
        return self.calculate_perspective_matrix(left, left + width, top, top - height, near, self.far)

    @staticmethod
    def calculate_perspective_matrix(left: float, right: float, top: float, bottom: float, near: float, far: float) -> Mat4:
        x = 2 * near / (right - left)
        y = 2 * near / (top - bottom)
        a = (right + left) / (right - left)
        b = (top + bottom) / (top - bottom)
        c = - (far + near) / (far - near)
        d = - 2 * far * near / (far - near)
        return np.array([
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [a, b, c, -1],
            [0, 0, d, 0]
        ])
