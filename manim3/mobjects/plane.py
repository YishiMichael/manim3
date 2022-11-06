import numpy as np

from ..mobjects.parametric_surface import ParametricSurface
from ..typing import *


__all__ = ["Plane"]


class Plane(ParametricSurface):
    def __init__(
        self: Self,
        x_segments: int = 1,
        y_segments: int = 1,
        **kwargs
    ):
        super().__init__(
            lambda x, y: np.array((x, y, 0.0)),
            (-0.5, 0.5),
            (-0.5, 0.5),
            resolution=(x_segments, y_segments),
            **kwargs
        )
