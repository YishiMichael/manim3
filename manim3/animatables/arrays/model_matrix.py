from __future__ import annotations


from typing import Self

import numpy as np
from scipy.spatial.transform import Rotation

from ...constants.custom_typing import (
    NP_3f8,
    NP_44f8,
    NP_x3f8
)
from ...lazy.lazy import Lazy
from .animatable_array import AnimatableArray


class ModelMatrix(AnimatableArray[NP_44f8]):
    __slots__ = ()

    @Lazy.variable()
    @staticmethod
    def _array_() -> NP_44f8:
        return np.identity(4)

    @classmethod
    def _apply(
        cls: type[Self],
        matrix: NP_44f8,
        vector: NP_3f8
    ) -> NP_3f8:
        v = matrix @ np.append(vector, 1.0)
        w_component = v[-1]
        result = np.delete(v, -1)
        if not np.allclose(w_component, 1.0):
            result /= w_component
        return result

    @classmethod
    def _apply_multiple(
        cls: type[Self],
        matrix: NP_44f8,
        vectors: NP_x3f8
    ) -> NP_x3f8:
        v = matrix @ np.append(vectors.T, np.ones((1, len(vectors))), axis=0)
        w_component = v[-1]
        result = np.delete(v, -1, axis=0)
        if not np.allclose(w_component, 1.0):
            result /= w_component
        return result.T

    @classmethod
    def _matrix_from_shift(
        cls: type[Self],
        vector: NP_3f8
    ) -> NP_44f8:
        matrix = np.identity(4)
        matrix[:3, 3] = vector
        return matrix

    @classmethod
    def _matrix_from_scale(
        cls: type[Self],
        factor: NP_3f8
    ) -> NP_44f8:
        matrix = np.identity(4)
        matrix[:3, :3] = np.diag(factor)
        return matrix

    @classmethod
    def _matrix_from_rotate(
        cls: type[Self],
        rotvec: NP_3f8
    ) -> NP_44f8:
        matrix = np.identity(4)
        matrix[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
        return matrix
