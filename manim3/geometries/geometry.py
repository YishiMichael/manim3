__all__ = [
    "Geometry",
    "GeometryData"
]


from dataclasses import dataclass

import moderngl
import numpy as np

from ..custom_typing import (
    Vec2sT,
    Vec3sT,
    VertexIndexType
)
from ..lazy.core import LazyObject
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..rendering.gl_buffer import (
    AttributesBuffer,
    IndexBuffer
)
from ..rendering.vertex_array import IndexedAttributesBuffer


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class GeometryData:
    index: VertexIndexType
    position: Vec3sT
    normal: Vec3sT
    uv: Vec2sT


class Geometry(LazyObject):
    __slots__ = ()

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _geometry_data_(cls) -> GeometryData:
        return GeometryData(
            index=np.zeros((0,), dtype=np.uint32),
            position=np.zeros((0, 3)),
            normal=np.zeros((0, 3)),
            uv=np.zeros((0, 2))
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _indexed_attributes_buffer_(
        cls,
        geometry_data: GeometryData
    ) -> IndexedAttributesBuffer:
        return IndexedAttributesBuffer(
            attributes_buffer=AttributesBuffer(
                fields=[
                    "vec3 in_position",
                    "vec3 in_normal",
                    "vec2 in_uv"
                ],
                num_vertex=len(geometry_data.position),
                data={
                    "in_position": geometry_data.position,
                    "in_normal": geometry_data.normal,
                    "in_uv": geometry_data.uv
                }
            ),
            index_buffer=IndexBuffer(
                data=geometry_data.index
            ),
            mode=moderngl.TRIANGLES
        )
