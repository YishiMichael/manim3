__all__ = ["ParametricSurfaceGeometry"]


from typing import Callable

import numpy as np

from ..custom_typing import Vec3T
from ..geometries.geometry import (
    Geometry,
    GeometryData
)


class ParametricSurfaceGeometry(Geometry):
    __slots__ = ()

    def __init__(
        self,
        *,
        func: Callable[[float, float], Vec3T],
        normal_func: Callable[[float, float], Vec3T],
        u_range: tuple[float, float],
        v_range: tuple[float, float],
        resolution: tuple[int, int] = (100, 100)
    ) -> None:
        u_start, u_stop = u_range
        v_start, v_stop = v_range
        u_len = resolution[0] + 1
        v_len = resolution[1] + 1
        index_grid = np.mgrid[0:u_len, 0:v_len]
        ne = index_grid[:, +1:, +1:]
        nw = index_grid[:, :-1, +1:]
        sw = index_grid[:, :-1, :-1]
        se = index_grid[:, +1:, :-1]
        index = np.ravel_multi_index(
            tuple(np.stack((se, sw, ne, sw, nw, ne), axis=3)),
            (u_len, v_len)
        ).flatten().astype(np.uint)

        uv = np.stack(np.meshgrid(
            np.linspace(0.0, 1.0, u_len),
            np.linspace(0.0, 1.0, v_len),
            indexing="ij"
        ), 2).reshape((-1, 2))
        samples = np.stack(np.meshgrid(
            np.linspace(u_start, u_stop, u_len),
            np.linspace(v_start, v_stop, v_len),
            indexing="ij"
        ), 2).reshape((-1, 2))
        position = np.apply_along_axis(lambda p: func(*p), 1, samples)
        normal = np.apply_along_axis(lambda p: normal_func(*p), 1, samples)

        super().__init__()
        self._geometry_data_ = GeometryData(
            index=index,
            position=position,
            normal=normal,
            uv=uv
        )
