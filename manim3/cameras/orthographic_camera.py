import pyrr

from ..cameras.camera import Camera
from ..constants import CAMERA_FAR, CAMERA_NEAR
from ..constants import FRAME_X_RADIUS, FRAME_Y_RADIUS
from ..typing import *


__all__ = ["OrthographicCamera"]


class OrthographicCamera(Camera):
    def __init__(
        self: Self,
        left: float = -FRAME_X_RADIUS,
        right: float = FRAME_X_RADIUS,
        top: float = FRAME_Y_RADIUS,
        bottom: float = -FRAME_Y_RADIUS,
        near: float = CAMERA_NEAR,
        far: float = CAMERA_FAR
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
