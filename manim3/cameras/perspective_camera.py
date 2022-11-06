import pyrr

from ..cameras.camera import Camera
from ..typing import *


__all__ = ["PerspectiveCamera"]


class PerspectiveCamera(Camera):
    def __init__(
        self: Self,
        fovy: float = 50.0,
        aspect: float = 1.0,
        near: float = 0.1,
        far: float = 100.0
    ):
        super().__init__()
        self.fovy: float = fovy
        self.aspect: float = aspect
        self.near: float = near
        self.far: float = far

    def get_projection_matrix(self: Self) -> pyrr.Matrix44:
        return pyrr.Matrix44.perspective_projection(
            self.fovy,
            self.aspect,
            self.near,
            self.far
        )
