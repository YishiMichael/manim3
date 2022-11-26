import numpy as np
import skia

from ..geometries.parametric_surface_geometry import ParametricSurfaceGeometry
from ..custom_typing import *


__all__ = [
    "FrameGeometry"
]


class FrameGeometry(ParametricSurfaceGeometry):
    def __init__(
        self,
        rect: skia.Rect,
        width_segments: int = 1,
        height_segments: int = 1
    ):
        super().__init__(
            lambda x, y: np.array((x, y, 0.0)),
            (rect.left(), rect.right()),
            (-rect.top(), -rect.bottom()),
            (width_segments, height_segments)
        )
