from __future__ import annotations

import math

from cameras.camera import Camera
from utils.arrays import Mat4


class PerspectiveCamera(Camera):
    def __init__(
        self,
        fov: float = 50.0,
        aspect: float = 1.0,
        near: float = 0.1,
        far: float = 2000.0
    ):
        super().__init__(near, far)
        #self._matrixWorldInverse: Mat4 = Mat4() # ~self.projectionMatrix

        self.fov: float = fov
        self.aspect: float = aspect  # width / height

        self.zoom: float = 1.0

        #self.focus: float = 10.0


        self.filmGauge: float = 540.0    # width of the film (default in millimeters)
        self.filmOffset: float = 0.0

        self.update_projection_matrix()

    def update_projection_matrix(self):
        DEGREES = math.pi / 180.0
        near = self.near
        top = near * math.tan(0.5 * self.fov * DEGREES) / self.zoom
        height = 2 * top
        width = self.aspect * height
        left = - 0.5 * width + self.filmOffset / self.filmGauge
        matrix = self.get_perspective_matrix(left, left + width, top, top - height, near, self.far)
        self._projection_matrix = matrix
        #self._matrixWorldInverse = ~matrix
        return self

    @staticmethod
    def get_perspective_matrix(left: float, right: float, top: float, bottom: float, near: float, far: float) -> Mat4:
        x = 2 * near / (right - left)
        y = 2 * near / (top - bottom)
        a = (right + left) / (right - left)
        b = (top + bottom) / (top - bottom)
        c = - (far + near) / (far - near)
        d = - 2 * far * near / (far - near)
        return Mat4(
              x, 0.0, 0.0,  0.0,
            0.0,   y, 0.0,  0.0,
              a,   b,   c, -1.0,
            0.0, 0.0,   d,  0.0
        )
