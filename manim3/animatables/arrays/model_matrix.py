from __future__ import annotations


from typing import Self

import numpy as np

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
        tx, ty, tz = vector
        return np.array((
            (1.0, 0.0, 0.0,  tx),
            (0.0, 1.0, 0.0,  ty),
            (0.0, 0.0, 1.0,  tz),
            (0.0, 0.0, 0.0, 1.0)
        ))

    @classmethod
    def _matrix_from_scale(
        cls: type[Self],
        factor: NP_3f8
    ) -> NP_44f8:
        sx, sy, sz = factor
        return np.array((
            ( sx, 0.0, 0.0, 0.0),
            (0.0,  sy, 0.0, 0.0),
            (0.0, 0.0,  sz, 0.0),
            (0.0, 0.0, 0.0, 1.0)
        ))

    @classmethod
    def _matrix_from_rotate(
        cls: type[Self],
        rotvec: NP_3f8
    ) -> NP_44f8:
        theta = np.linalg.norm(rotvec)
        if theta != 0.0:
            s = np.sin(theta) / theta
            c = (1.0 - np.cos(theta)) / (theta * theta)
        else:
            s = 1.0
            c = 0.5
        rx, ry, rz = rotvec
        return np.array((
            (1.0 - c * (ry * ry + rz * rz),          c * rx * ry - s * rz,          c * rx * rz + s * ry, 0.0),
            (         c * ry * rx + s * rz, 1.0 - c * (rx * rx + rz * rz),          c * ry * rz - s * rx, 0.0),
            (         c * rz * rx - s * ry,          c * rz * ry + s * rx, 1.0 - c * (rx * rx + ry * ry), 0.0),
            (                          0.0,                           0.0,                           0.0, 1.0)
        ))
