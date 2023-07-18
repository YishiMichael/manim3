from ....constants.custom_typing import (
    NP_3f8,
    NP_44f8
)
from .model_interpolant import ModelInterpolant


class ConstantModelInterpolant(ModelInterpolant):
    __slots__ = ("_matrix",)

    def __init__(
        self,
        matrix: NP_44f8
    ) -> None:
        super().__init__()
        self._matrix: NP_44f8 = matrix

    def __call__(
        self,
        alpha: float | NP_3f8 = 1.0
    ) -> NP_44f8:
        return self._matrix
