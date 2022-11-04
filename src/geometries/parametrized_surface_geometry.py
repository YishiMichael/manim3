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

    def get_indices(self: Self) -> VertexIndicesType:
        u_len = len(self.u_samples)
        v_len = len(self.v_samples)
        uv_grid = np.mgrid[0:u_len, 0:v_len]
        a = uv_grid[:, :-1, :-1]
        b = uv_grid[:, +1:, :-1]
        c = uv_grid[:, :-1, +1:]
        d = uv_grid[:, +1:, +1:]
        return np.ravel_multi_index(
            np.stack((b, a, d, a, c, d), axis=-1),
            (u_len, v_len)
        ).flatten()#.astype(np.uint32)  # TODO: convert type

    def get_vertex_attributes(self: Self) -> AttributesType:
        func = self.func
        vertices = np.array([
            func(u, v)
            for v in self.v_samples
            for u in self.u_samples
        ])
        #x, y = np.meshgrid(
        #    np.linspace(-1.0, 1.0, self.x_segments + 1),
        #    np.linspace(-1.0, 1.0, self.y_segments + 1)
        #)
        #vertices = np.stack(
        #    (x, y, np.zeros_like(x)),
        #    axis=-1
        #).reshape((-1, 3))
        result = np.zeros(len(vertices), dtype=[
            ("in_position", np.float32, (3,))  # TODO: convert type
        ])
        result["in_position"] = vertices
        return result
