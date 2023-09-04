import numpy as np

from ....constants.custom_typing import (
    NP_3f8,
    NP_44f8
)
from .remodel_handler import RemodelHandler


class ShiftRemodelHandler(RemodelHandler):
    __slots__ = ("_vector",)

    def __init__(
        self,
        vector: NP_3f8
    ) -> None:
        super().__init__()
        self._vector: NP_3f8 = vector

    def _remodel(
        self,
        alpha: float | NP_3f8 = 1.0
    ) -> NP_44f8:
        m = np.identity(4)
        m[:3, 3] = self._vector * alpha
        return m
