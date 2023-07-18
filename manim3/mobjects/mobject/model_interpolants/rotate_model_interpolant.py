import numpy as np
from scipy.spatial.transform import Rotation

from ....constants.custom_typing import (
    NP_3f8,
    NP_44f8
)
from ....utils.space import (
    Lerp,
    SpaceUtils
)
from .model_interpolant import ModelInterpolant


class RotateModelInterpolant(ModelInterpolant):
    __slots__ = ("_rotvec_lerp",)

    def __init__(
        self,
        rotvec: NP_3f8
    ) -> None:
        super().__init__()
        self._rotvec_lerp: Lerp[NP_3f8] = SpaceUtils.lerp(np.zeros((3,)), rotvec)

    def __call__(
        self,
        alpha: float | NP_3f8 = 1.0
    ) -> NP_44f8:
        m = np.identity(4)
        m[:3, :3] = Rotation.from_rotvec(self._rotvec_lerp(alpha)).as_matrix()
        return m
