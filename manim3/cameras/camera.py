from abc import ABC, abstractmethod

import numpy as np
import pyrr

from ..typing import *


__all__ = ["Camera"]


class Camera(ABC):
    def __init__(self: Self):
        #self.view_matrix: pyrr.Matrix44 = pyrr.Matrix44.look_at(np.array([0.0, 5.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))  # TODO
        self.view_matrix: pyrr.Matrix44 = pyrr.Matrix44.look_at(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))  # TODO

    @abstractmethod
    def get_projection_matrix(self: Self) -> pyrr.Matrix44:
        raise NotImplementedError

    def get_view_matrix(self: Self) -> pyrr.Matrix44:
        return self.view_matrix
