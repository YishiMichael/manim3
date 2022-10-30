from dataclasses import dataclass
from typing import Self

import numpy as np

from cameras.camera import Camera
from utils.arrays import Mat4


@dataclass
class OrthographicCamera(Camera):
    left: float = -1.0
    right: float = 1.0
    top: float = 1.0
    bottom: float = -1.0
    near: float = 0.1
    far: float = 2000.0

    def calculate_projection_matrix(self: Self) -> Mat4:
        dx = (self.right - self.left) / (2 * self.zoom)
        dy = (self.top - self.bottom) / (2 * self.zoom)
        cx = (self.right + self.left) / 2
        cy = (self.top + self.bottom) / 2

        left = cx - dx
        right = cx + dx
        top = cy + dy
        bottom = cy - dy

        return self.calculate_orthographic_matrix(left, right, top, bottom, self.near, self.far)

    @staticmethod
    def calculate_orthographic_matrix(left: float, right: float, top: float, bottom: float, near: float, far: float) -> Mat4:
        w = 1.0 / (right - left)
        h = 1.0 / (top - bottom)
        p = 1.0 / (far - near)
        x = (right + left) * w
        y = (top + bottom) * h
        z = (far + near) * p
        return np.array([
            [2 * w, 0, 0, 0],
            [0, 2 * h, 0, 0],
            [0, 0, -2 * p, 0],
            [-x, -y, -z, 1]
        ])
