import numpy as np

from ...constants.constants import ORIGIN
from ...constants.custom_typing import NP_x3f8
from ...lazy.lazy import Lazy
from .model import Model


class Point(Model):
    __slots__ = ()

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _local_sample_positions_() -> NP_x3f8:
        return np.array((ORIGIN,))
