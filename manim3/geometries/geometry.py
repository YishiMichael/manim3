__all__ = ["Geometry"]


from ..custom_typing import (
    Vec2sT,
    Vec3sT,
    VertexIndexType
)
from ..utils.lazy import (
    LazyBase,
    lazy_property,
    lazy_property_writable
)
from ..utils.render_procedure import (
    AttributesBuffer,
    IndexBuffer
)


class Geometry(LazyBase):
    def __init__(
        self,
        index: VertexIndexType,
        position: Vec3sT,
        normal: Vec3sT,
        uv: Vec2sT
    ):
        super().__init__()
        self._index_ = index
        self._position_ = position
        self._normal_ = normal
        self._uv_ = uv

    @lazy_property_writable
    @staticmethod
    def _index_() -> VertexIndexType:
        return NotImplemented

    @lazy_property_writable
    @staticmethod
    def _position_() -> Vec3sT:
        return NotImplemented

    @lazy_property_writable
    @staticmethod
    def _normal_() -> Vec3sT:
        return NotImplemented

    @lazy_property_writable
    @staticmethod
    def _uv_() -> Vec2sT:
        return NotImplemented

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
        position: Vec3sT,
        normal: Vec3sT,
        uv: Vec2sT
    ) -> AttributesBuffer:
        return attributes_o.write({
            "in_position": position,
            "in_normal": normal,
            "in_uv": uv
        })

    @lazy_property
    @staticmethod
    def _index_buffer_o_() -> IndexBuffer:
        return IndexBuffer()

    @lazy_property
    @staticmethod
    def _index_buffer_(
        index_buffer_o: IndexBuffer,
        index: VertexIndexType
    ) -> IndexBuffer:
        return index_buffer_o.write(index)
