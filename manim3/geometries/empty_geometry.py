__all__ = ["EmptyGeometry"]


import numpy as np

from ..geometries.geometry import Geometry


class EmptyGeometry(Geometry):
    def __init__(self):
        super().__init__(
            index=np.zeros((0,), dtype=np.uint32),
            position=np.zeros((0, 3)),
            normal=np.zeros((0, 3)),
            uv=np.zeros((0, 2))
        )
