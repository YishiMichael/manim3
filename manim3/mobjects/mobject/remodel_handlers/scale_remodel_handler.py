import numpy as np

from ....constants.custom_typing import (
    NP_3f8,
    NP_44f8
)
from ....utils.space import SpaceUtils
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
        self._factor: NP_3f8 = factor

    def remodel(
        self,
        alpha: float | NP_3f8 = 1.0
    ) -> NP_44f8:
        factor = SpaceUtils.lerp(np.ones((3,)), self._factor, alpha)
        m = np.identity(4)
        m[:3, :3] = np.diag(factor)
        return m
