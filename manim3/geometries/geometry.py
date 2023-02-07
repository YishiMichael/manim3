__all__ = [
    "GeometryData",
    "Geometry"
]


from dataclasses import dataclass

import numpy as np

from ..custom_typing import (
    Vec2sT,
    Vec3sT,
    VertexIndexType
)
from ..rendering.render_procedure import (
    AttributesBuffer,
    IndexBuffer
)
from ..utils.lazy import (
    LazyBase,
    lazy_basedata,
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


class Geometry(LazyBase):
    @lazy_basedata
    @staticmethod
    def _geometry_data_() -> GeometryData:
        return GeometryData(
            index=np.zeros((0,), dtype=np.uint32),
            position=np.zeros((0, 3)),
            normal=np.zeros((0, 3)),
            uv=np.zeros((0, 2))
        )

    #@lazy_property
    #@staticmethod
    #def _attributes_o_() -> AttributesBuffer:
    #    return AttributesBuffer([
    #        "vec3 in_position",
    #        "vec3 in_normal",
    #        "vec2 in_uv"
    #    ])

    @lazy_property
    @staticmethod
    def _attributes_(
        #attributes_o: AttributesBuffer,
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

    #@lazy_property
    #@staticmethod
    #def _index_buffer_o_() -> IndexBuffer:
    #    return IndexBuffer()

    @lazy_property
    @staticmethod
    def _index_buffer_(
        #index_buffer_o: IndexBuffer,
        geometry_data: GeometryData
    ) -> IndexBuffer:
        return IndexBuffer(
            data=geometry_data.index
        )
