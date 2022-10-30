from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from utils.arrays import Mat4


class Camera(ABC):
    def __init__(self):
        self.__projection_matrix: Mat4 = np.eye(4)  # TODO

    def get_projection_matrix(self: Self) -> Mat4:
        return self.__projection_matrix

    def update_projection_matrix(self: Self) -> Self:
        self.__projection_matrix = self.calculate_projection_matrix()
        return self

    @abstractmethod
    def calculate_projection_matrix(self: Self) -> Mat4:
        raise NotImplementedError
