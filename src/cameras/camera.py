from abc import ABC, abstractmethod

import numpy as np
import pyrr

from utils.typing import *


__all__ = ["Camera"]


class Camera(ABC):
    def __init__(self: Self):
        self.matrix: pyrr.Matrix44 = pyrr.Matrix44.look_at(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))

    @abstractmethod
    def get_projection_matrix(self: Self) -> pyrr.Matrix44:
        raise NotImplementedError

    def get_transform_matrix(self: Self) -> pyrr.Matrix44:
        return self.get_projection_matrix() @ ~self.matrix
