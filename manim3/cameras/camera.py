from abc import ABC, abstractmethod

import pyrr

from ..constants import CAMERA_ALTITUDE
from ..constants import ORIGIN, OUT, UP
from ..typing import *


__all__ = ["Camera"]


class Camera(ABC):
    def __init__(self: Self):
        self.view_matrix: pyrr.Matrix44 = pyrr.Matrix44.look_at(CAMERA_ALTITUDE * OUT, ORIGIN, UP)

    @abstractmethod
    def get_projection_matrix(self: Self) -> pyrr.Matrix44:
        raise NotImplementedError

    def get_view_matrix(self: Self) -> pyrr.Matrix44:
        return self.view_matrix
