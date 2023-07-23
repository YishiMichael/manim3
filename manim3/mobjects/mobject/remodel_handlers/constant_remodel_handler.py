from ....constants.custom_typing import (
    NP_3f8,
    NP_44f8
)
from .remodel_handler import RemodelHandler


class ConstantRemodelHandler(RemodelHandler):
    __slots__ = ("_matrix",)

    def __init__(
        self,
        matrix: NP_44f8
    ) -> None:
        super().__init__()
        self._matrix: NP_44f8 = matrix

    def remodel(
        self,
        alpha: float | NP_3f8 = 1.0
    ) -> NP_44f8:
        return self._matrix
