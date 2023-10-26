from __future__ import annotations


from typing import Self

import numpy as np

from ...constants.constants import ORIGIN
from ...constants.custom_typing import (
    NP_3f8,
    NP_x3f8
)
from ...lazy.lazy import Lazy
from .model import Model


class Point(Model):
    __slots__ = ()

    def __init__(
        self: Self,
        position: NP_3f8 | None = None
    ) -> None:
        super().__init__()
        if position is not None:
            self.shift(position)

    @Lazy.property()
    @staticmethod
    def _local_sample_positions_() -> NP_x3f8:
        return np.array((ORIGIN,))
