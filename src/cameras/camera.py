from __future__ import annotations

from mobject import Mobject
from utils.arrays import Mat4


class Camera(Mobject):
    def __init__(self, near: float, far: float):
        super().__init__()
        self.near: float = near
        self.far: float = far
        self.projection_matrix: Mat4 = Mat4()  # TODO

    def update_projection_matrix(self):
        raise NotImplementedError
