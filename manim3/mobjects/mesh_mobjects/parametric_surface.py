from __future__ import annotations


from typing import (
    Callable,
    Self
)

import numpy as np

from ...animatables.mesh import Mesh
from ...constants.custom_typing import (
    NP_x2f8,
    NP_x3f8
)
from .mesh_mobject import MeshMobject


class ParametricSurface(MeshMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        *,
        func: Callable[[NP_x2f8], NP_x3f8],
        normal_func: Callable[[NP_x2f8], NP_x3f8],
        u_range: tuple[float, float],
        v_range: tuple[float, float],
        resolution: tuple[int, int] = (128, 128)
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
        faces = np.ravel_multi_index(
            tuple(np.stack((se, sw, ne, sw, nw, ne), axis=3)),
            (u_len, v_len)
        ).reshape((-1, 3))

        uvs = np.stack(np.meshgrid(
            np.linspace(0.0, 1.0, u_len),
            np.linspace(0.0, 1.0, v_len),
            indexing="ij"
        ), 2).reshape((-1, 2))
        samples: NP_x2f8 = np.stack(np.meshgrid(
            np.linspace(u_start, u_stop, u_len),
            np.linspace(v_start, v_stop, v_len),
            indexing="ij"
        ), 2).reshape((-1, 2))
        positions = func(samples)
        normals = normal_func(samples)

        super().__init__(Mesh(
            positions=positions,
            normals=normals,
            uvs=uvs,
            faces=faces
        ))
