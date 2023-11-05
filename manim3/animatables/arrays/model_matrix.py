from __future__ import annotations


import numpy as np

from ...constants.custom_typing import NP_44f8
from ...lazy.lazy import Lazy
from .animatable_array import AnimatableArray


class ModelMatrix(AnimatableArray[NP_44f8]):
    __slots__ = ()

    @Lazy.variable()
    @staticmethod
    def _array_() -> NP_44f8:
        return np.identity(4)
