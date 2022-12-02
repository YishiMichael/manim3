__all__ = ["Camera"]


from abc import ABC, abstractmethod

import pyrr

from ..constants import CAMERA_ALTITUDE
from ..constants import ORIGIN, OUT, UP
from ..custom_typing import *


class Camera(ABC):
    def __init__(self):
        self.view_matrix: Matrix44Type = pyrr.matrix44.create_look_at(CAMERA_ALTITUDE * OUT, ORIGIN, UP)

    @abstractmethod
    def get_projection_matrix(self) -> Matrix44Type:
        raise NotImplementedError

    def get_view_matrix(self) -> Matrix44Type:
        return self.view_matrix
