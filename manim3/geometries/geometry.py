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
from ..rendering.glsl_buffers import (
    AttributesBuffer,
    IndexBuffer
)
from ..rendering.vertex_array import VertexArray
from ..utils.lazy import (
    LazyObject,
    lazy_object_raw,
    lazy_property
)


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

    @lazy_object_raw
    @staticmethod
    def _geometry_data_() -> GeometryData:
        return GeometryData(
            index=np.zeros((0,), dtype=np.uint32),
            position=np.zeros((0, 3)),
            normal=np.zeros((0, 3)),
            uv=np.zeros((0, 2))
        )

    @lazy_property
    @staticmethod
    def _attributes_(
        geometry_data: GeometryData
    ) -> AttributesBuffer:
        return AttributesBuffer(
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
        )

    @lazy_property
    @staticmethod
    def _index_buffer_(
        geometry_data: GeometryData
    ) -> IndexBuffer:
        return IndexBuffer(
            data=geometry_data.index
        )

    @lazy_property
    @staticmethod
    def _vertex_array_(
        attributes: AttributesBuffer,
        index_buffer: IndexBuffer
    ) -> VertexArray:
        return VertexArray(
            attributes=attributes,
            index_buffer=index_buffer,
            mode=moderngl.TRIANGLES
        )
