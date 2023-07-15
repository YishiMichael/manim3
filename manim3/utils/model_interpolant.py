from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from ..constants.custom_typing import (
    NP_3f8,
    NP_44f8
)
from ..utils.space import SpaceUtils


class ModelInterpolant:
    __slots__ = ("_matrix_callback",)

    def __init__(
        self,
        callback: Callable[[float | NP_3f8], NP_44f8]
    ) -> None:
        super().__init__()
        self._matrix_callback: Callable[[float | NP_3f8], NP_44f8] = callback

    def __call__(
        self,
        alpha: float | NP_3f8 = 1.0
    ) -> NP_44f8:
        return self._matrix_callback(alpha)

    @classmethod
    def from_shift(
        cls,
        vector: NP_3f8
    ) -> "ModelInterpolant":
        lerp_callback = SpaceUtils.lerp(np.zeros(()), vector)

        def callback(
            alpha: float | NP_3f8
        ) -> NP_44f8:
            m = np.identity(4)
            m[:3, 3] = lerp_callback(alpha)
            return m

        return ModelInterpolant(callback)

    @classmethod
    def from_scale(
        cls,
        factor: float | NP_3f8
    ) -> "ModelInterpolant":
        if not isinstance(factor, np.ndarray):
            factor *= np.ones((3,))
        lerp_callback = SpaceUtils.lerp(np.ones(()), factor)

        def callback(
            alpha: float | NP_3f8
        ) -> NP_44f8:
            m = np.identity(4)
            m[:3, :3] = np.diag(lerp_callback(alpha))
            return m

        return ModelInterpolant(callback)

    @classmethod
    def from_rotate(
        cls,
        rotvec: NP_3f8
    ) -> "ModelInterpolant":
        lerp_callback = SpaceUtils.lerp(np.zeros(()), rotvec)

        def callback(
            alpha: float | NP_3f8
        ) -> NP_44f8:
            m = np.identity(4)
            m[:3, :3] = Rotation.from_rotvec(lerp_callback(alpha)).as_matrix()
            return m

        return ModelInterpolant(callback)
