__all__ = ["ShapeGeometry"]


import numpy as np

from ..geometries.geometry import Geometry
from ..utils.shape import Shape


class ShapeGeometry(Geometry):
    def __init__(
        self,
        shape: Shape
    ):
        coords, index = shape._triangulation_
        position = np.insert(coords, 2, 0.0, axis=1)
        normal = np.repeat(np.array((0.0, 0.0, 1.0))[None], len(position), axis=0)
        super().__init__(
            index=index,
            position=position,
            normal=normal,
            uv=coords
        )
