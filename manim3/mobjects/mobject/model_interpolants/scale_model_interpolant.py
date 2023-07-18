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


class ScaleModelInterpolant(ModelInterpolant):
    __slots__ = ("_factor_lerp",)

    def __init__(
        self,
        factor: float | NP_3f8
    ) -> None:
        super().__init__()
        if not isinstance(factor, np.ndarray):
            factor *= np.ones((3,))
        self._factor_lerp: Lerp[NP_3f8] = SpaceUtils.lerp(np.zeros((3,)), factor)

    def __call__(
        self,
        alpha: float | NP_3f8 = 1.0
    ) -> NP_44f8:
        m = np.identity(4)
        m[:3, :3] = np.diag(self._factor_lerp(alpha))
        return m
