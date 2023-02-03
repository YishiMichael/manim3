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
    @lazy_property
    @staticmethod
    def _data_() -> GeometryData:
        return GeometryData(
            index=np.zeros((0,), dtype=np.uint32),
            position=np.zeros((0, 3)),
            normal=np.zeros((0, 3)),
            uv=np.zeros((0, 2))
        )

    @lazy_property
    @staticmethod
    def _attributes_o_() -> AttributesBuffer:
        return AttributesBuffer([
            "vec3 in_position",
            "vec3 in_normal",
            "vec2 in_uv"
        ])

    @lazy_property
    @staticmethod
    def _attributes_(
        attributes_o: AttributesBuffer,
        data: GeometryData
    ) -> AttributesBuffer:
        return attributes_o.write({
            "in_position": data.position,
            "in_normal": data.normal,
            "in_uv": data.uv
        })

    @lazy_property
    @staticmethod
    def _index_buffer_o_() -> IndexBuffer:
        return IndexBuffer()

    @lazy_property
    @staticmethod
    def _index_buffer_(
        index_buffer_o: IndexBuffer,
        data: GeometryData
    ) -> IndexBuffer:
        return index_buffer_o.write(data.index)
