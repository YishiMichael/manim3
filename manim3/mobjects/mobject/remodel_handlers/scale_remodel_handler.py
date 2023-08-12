import numpy as np

from ....constants.custom_typing import (
    NP_3f8,
    NP_44f8
)
from .remodel_handler import RemodelHandler


class ScaleRemodelHandler(RemodelHandler):
    __slots__ = ("_factor",)

    def __init__(
        self,
        factor: float | NP_3f8
    ) -> None:
        super().__init__()
        if not isinstance(factor, np.ndarray):
            factor *= np.ones((3,))
        assert (factor > 0.0).all(), "Scale factor must be positive"
        self._factor: NP_3f8 = factor

    def remodel(
        self,
        alpha: float | NP_3f8 = 1.0
    ) -> NP_44f8:
        m = np.identity(4)
        m[:3, :3] = np.diag(self._factor ** alpha)
        return m
