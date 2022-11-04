import pyrr

from cameras.camera import Camera
from utils.typing import *


__all__ = ["OrthographicCamera"]


class OrthographicCamera(Camera):
    def __init__(
        self: Self,
        left: float = -1.0,
        right: float = 1.0,
        top: float = 1.0,
        bottom: float = -1.0,
        near: float = 0.1,
        far: float = 100.0
    ):
        super().__init__()
        self.left: float = left
        self.right: float = right
        self.top: float = top
        self.bottom: float = bottom
        self.near: float = near
        self.far: float = far

    def get_projection_matrix(self: Self) -> pyrr.Matrix44:
        return pyrr.Matrix44.orthogonal_projection(
            self.left,
            self.right,
            self.top,
            self.bottom,
            self.near,
            self.far
        )
