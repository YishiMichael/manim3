import numpy as np
from typing import Callable

from ..geometries.geometry import Geometry, GeometryAttributes
from ..custom_typing import *


__all__ = ["ParametricSurfaceGeometry"]


class ParametricSurfaceGeometry(Geometry):
    def __init__(
        self,
        func: Callable[[float, float], Vector3Type],
        u_range: tuple[Real, Real],
        v_range: tuple[Real, Real],
        resolution: tuple[int, int] = (100, 100)
    ):
        self.func: Callable[[float, float], Vector3Type] = func
        self.u_range: tuple[Real, Real] = u_range
        self.v_range: tuple[Real, Real] = v_range
        self.resolution: tuple[int, int] = resolution
        super().__init__()

    def init_geometry_attributes(self) -> GeometryAttributes:
        u_start, u_stop = self.u_range
        v_start, v_stop = self.v_range
        u_len = self.resolution[0] + 1
        v_len = self.resolution[1] + 1
        index_grid = np.mgrid[0:u_len, 0:v_len]
        ne = index_grid[:, +1:, +1:]
        nw = index_grid[:, :-1, +1:]
        sw = index_grid[:, :-1, :-1]
        se = index_grid[:, +1:, :-1]
        indices = np.ravel_multi_index(
            tuple(np.stack((se, sw, ne, sw, nw, ne), axis=3)),
            (u_len, v_len)
        ).flatten().astype(np.int32)

        uvs = np.stack(np.meshgrid(
            np.linspace(0.0, 1.0, u_len),
            np.linspace(0.0, 1.0, v_len),
            indexing="ij"
        ), 2).reshape((-1, 2))
        samples = np.stack(np.meshgrid(
            np.linspace(u_start, u_stop, u_len),
            np.linspace(v_start, v_stop, v_len),
            indexing="ij"
        ), 2).reshape((-1, 2))
        positions = np.apply_along_axis(lambda p: self.func(*p), 1, samples)
        return GeometryAttributes(
            indices=indices,
            positions=positions,
            uvs=uvs
        )
        # TODO: normals using `from scipy.misc import derivative`
