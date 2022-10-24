from __future__ import annotations

from cameras.camera import Camera
from utils.arrays import Mat4


class OrthographicCamera(Camera):
    def __init__(
        self,
        left: float = -1.0,
        right: float = 1.0,
        top: float = 1.0,
        bottom: float = -1.0,
        near: float = 0.1,
        far: float = 2000.0
    ):
        super().__init__(near, far)
        #self._matrixWorldInverse: Mat4 = Mat4() # ~self.projectionMatrix

        self.left: float = left
        self.right: float = right
        self.top: float = top
        self.bottom: float = bottom

        self.zoom: float = 1.0

        self.update_projection_matrix()

    def update_projection_matrix(self):
        dx = (self.right - self.left) / (2 * self.zoom)
        dy = (self.top - self.bottom) / (2 * self.zoom)
        cx = (self.right + self.left) / 2
        cy = (self.top + self.bottom) / 2

        left = cx - dx
        right = cx + dx
        top = cy + dy
        bottom = cy - dy

        matrix = self.get_orthographic_matrix(left, right, top, bottom, self.near, self.far)
        self.projection_matrix = matrix
        #self._matrixWorldInverse = ~matrix
        return self

    @staticmethod
    def get_orthographic_matrix(left: float, right: float, top: float, bottom: float, near: float, far: float) -> Mat4:
        w = 1.0 / ( right - left)
        h = 1.0 / ( top - bottom)
        p = 1.0 / ( far - near)
        x = (right + left) * w
        y = (top + bottom) * h
        z = (far + near) * p
        return Mat4(
            2 * w,   0.0,    0.0, 0.0,
              0.0, 2 * h,    0.0, 0.0,
              0.0,   0.0, -2 * p, 0.0,
               -x,    -y,     -z, 1.0
        )
