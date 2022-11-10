import numpy as np
import pyrr

from ..cameras.camera import Camera
from ..constants import ASPECT_RATIO, FRAME_Y_RADIUS
from ..constants import CAMERA_ALTITUDE, CAMERA_FAR, CAMERA_NEAR
from ..constants import DEGREES
from ..typing import *


__all__ = ["PerspectiveCamera"]


class PerspectiveCamera(Camera):
    def __init__(
        self: Self,
        fovy: Real = 2.0 * np.arctan(FRAME_Y_RADIUS / CAMERA_ALTITUDE) / DEGREES,
        aspect: Real = ASPECT_RATIO,
        near: Real = CAMERA_NEAR,
        far: Real = CAMERA_FAR
    ):
        super().__init__()
        self.fovy: float = float(fovy)
        self.aspect: float = float(aspect)
        self.near: float = float(near)
        self.far: float = float(far)

    def get_projection_matrix(self: Self) -> pyrr.Matrix44:
        return pyrr.Matrix44.perspective_projection(
            self.fovy,
            self.aspect,
            self.near,
            self.far
        )
