import numpy as np

from ...constants.custom_typing import (
    NP_3f8,
    NP_44f8,
    NP_x3f8
)
from ...lazy.lazy import Lazy
from ...lazy.lazy_object import LazyObject
from .animatable_array import AnimatableArray


class AffineApplier(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        matrix: NP_44f8
    ) -> None:
        super().__init__()
        self._matrix_ = matrix

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _matrix_() -> NP_44f8:
        return NotImplemented

    def apply(
        self,
        vector: NP_3f8
    ) -> NP_3f8:
        v = self._matrix_ @ np.append(vector, 1.0)
        w_component = v[-1]
        result = np.delete(v, -1)
        if not np.allclose(w_component, 1.0):
            result /= w_component
        return result

    def apply_multiple(
        self,
        vectors: NP_x3f8
    ) -> NP_x3f8:
        v = self._matrix_ @ np.append(vectors.T, np.ones((1, len(vectors))), axis=0)
        w_component = v[-1]
        result = np.delete(v, -1, axis=0)
        if not np.allclose(w_component, 1.0):
            result /= w_component
        return result.T


class ModelMatrix(AnimatableArray[NP_44f8]):
    __slots__ = ()

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _array_() -> NP_44f8:
        return np.identity(4)

    @Lazy.property()
    @staticmethod
    def _applier_(
        array: NP_44f8
    ) -> AffineApplier:
        return AffineApplier(array)

    @classmethod
    def _convert_input(
        cls,
        model_matrix_input: object
    ):
        raise ValueError("Cannot manually set the model matrix")