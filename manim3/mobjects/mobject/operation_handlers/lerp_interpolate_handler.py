from typing import TypeVar

import numpy as np

from ....utils.space import SpaceUtils
from .interpolate_handler import InterpolateHandler


_T = TypeVar("_T", bound=np.ndarray)


class LerpInterpolateHandler(InterpolateHandler[_T]):
    __slots__ = (
        "_tensor_0",
        "_tensor_1"
    )

    def __init__(
        self,
        src_0: _T,
        src_1: _T
    ) -> None:
        super().__init__(src_0, src_1)
        self._tensor_0: _T = src_0
        self._tensor_1: _T = src_1

    def interpolate(
        self,
        alpha: float
    ) -> _T:
        return SpaceUtils.lerp(self._tensor_0, self._tensor_1, alpha)
