from __future__ import annotations


from typing import Self

import numpy as np

from .parametric_surface import ParametricSurface


class Plane(ParametricSurface):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        super().__init__(
            func=lambda samples: np.concatenate((samples, np.zeros((len(samples), 1))), axis=1),
            normal_func=lambda samples: np.concatenate((np.zeros_like(samples), np.ones((len(samples), 1))), axis=1),
            u_range=(-1.0, 1.0),
            v_range=(-1.0, 1.0),
            resolution=(1, 1)
        )
