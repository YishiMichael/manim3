import pyrr

from ..cameras.camera import Camera
from ..constants import CAMERA_FAR, CAMERA_NEAR
from ..constants import FRAME_X_RADIUS, FRAME_Y_RADIUS
from ..custom_typing import *


__all__ = ["OrthographicCamera"]


class OrthographicCamera(Camera):
    def __init__(
        self: Self,
        left: Real = -FRAME_X_RADIUS,
        right: Real = FRAME_X_RADIUS,
        top: Real = FRAME_Y_RADIUS,
        bottom: Real = -FRAME_Y_RADIUS,
        near: Real = CAMERA_NEAR,
        far: Real = CAMERA_FAR
    ):
        super().__init__()
        self.left: float = float(left)
        self.right: float = float(right)
        self.top: float = float(top)
        self.bottom: float = float(bottom)
        self.near: float = float(near)
        self.far: float = float(far)

    def get_projection_matrix(self: Self) -> pyrr.Matrix44:
        return pyrr.Matrix44.orthogonal_projection(
            self.left,
            self.right,
            self.top,
            self.bottom,
            self.near,
            self.far
        )
