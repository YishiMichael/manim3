import numpy as np
from typing import Callable

from geometries.geometry import Geometry
from utils.typing import *


__all__ = ["ParametrizedSurfaceGeometry"]


class ParametrizedSurfaceGeometry(Geometry):
    def __init__(
        self: Self,
        func: Callable[[float, float], Vector3Type],
        u_samples: FloatArrayType,
        v_samples: FloatArrayType
    ):
        super().__init__()
        self.func: Callable[[float, float], Vector3Type] = func
        self.u_samples: FloatArrayType = u_samples
        self.v_samples: FloatArrayType = v_samples

    def get_vertex_indices(self: Self) -> VertexIndicesType:
        u_len = len(self.u_samples)
        v_len = len(self.v_samples)
        index_grid = np.mgrid[0:u_len, 0:v_len]
        ne = index_grid[:, +1:, +1:]
        nw = index_grid[:, :-1, +1:]
        sw = index_grid[:, :-1, :-1]
        se = index_grid[:, +1:, :-1]
        return np.ravel_multi_index(
            tuple(np.stack((se, sw, ne, sw, nw, ne), axis=-1)),
            (u_len, v_len)
        ).flatten().astype(np.int32)

    def get_attributes_v(self: Self) -> AttributesItemType:
        uv_grid = np.meshgrid(self.u_samples, self.v_samples, indexing="ij")
        return np.fromiter((
            (
                self.func(u, v),
                np.array([u, v])
                # derivatives for normal vectors...
            ) for u, v in np.nditer(uv_grid)
        ), dtype=np.dtype([
            ("in_position", np.float32, (3,)),
            ("in_uv", np.float32, (2,))
        ]))
