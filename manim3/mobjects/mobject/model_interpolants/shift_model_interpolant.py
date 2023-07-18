import numpy as np

from ....constants.custom_typing import (
    NP_3f8,
    NP_44f8
)
from ....utils.space import (
    Lerp,
    SpaceUtils
)
from .model_interpolant import ModelInterpolant


class ShiftModelInterpolant(ModelInterpolant):
    __slots__ = ("_vector_lerp",)

    def __init__(
        self,
        vector: NP_3f8
    ) -> None:
        super().__init__()
        self._vector_lerp: Lerp[NP_3f8] = SpaceUtils.lerp(np.zeros((3,)), vector)

    def __call__(
        self,
        alpha: float | NP_3f8 = 1.0
    ) -> NP_44f8:
        m = np.identity(4)
        m[:3, 3] = self._vector_lerp(alpha)
        return m
