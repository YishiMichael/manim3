from __future__ import annotations


from typing import Self

import numpy as np

from ...constants.constants import UP
from ...utils.space_utils import SpaceUtils
from .parametric_surface import ParametricSurface


class Plane(ParametricSurface):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        super().__init__(
            func=SpaceUtils.increase_dimension,
            normal_func=lambda samples: np.repeat((UP,), len(samples), axis=0),
            u_range=(-1.0, 1.0),
            v_range=(-1.0, 1.0),
            resolution=(1, 1)
        )
