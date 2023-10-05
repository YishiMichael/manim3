from __future__ import annotations


from typing import Self

import numpy as np

from ...constants.custom_typing import NP_3f8
from .polyline import Polyline


class Line(Polyline):
    __slots__ = ()

    def __init__(
        self: Self,
        position_0: NP_3f8,
        position_1: NP_3f8
    ) -> None:
        super().__init__(
            positions=np.array((position_0, position_1))
        )
