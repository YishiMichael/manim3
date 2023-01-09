__all__ = ["Geometry"]


import numpy as np

from ..custom_typing import (
    Vec2sT,
    Vec3sT,
    Vec4sT,
    Vec4T,
    VertexIndicesType
)
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)
from ..utils.renderable import (
    AttributesBuffer,
    IndexBuffer,
    Renderable
)


class Geometry(Renderable):
    def __init__(
        self,
        index: VertexIndicesType,
        position: Vec3sT,
        normal: Vec3sT,
        uv: Vec2sT
    ):
        super().__init__()
        self._index_ = index
        self._position_ = position
        self._normal_ = normal
        self._uv_ = uv

    @lazy_property_initializer_writable
    @staticmethod
    def _index_() -> VertexIndicesType:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _index_buffer_o_() -> IndexBuffer:
        return IndexBuffer()

    @lazy_property
    @staticmethod
    def _index_buffer_(
        index_buffer_o: IndexBuffer,
        index: VertexIndicesType
    ) -> IndexBuffer:
        index_buffer_o.write(index)
        return index_buffer_o

    @lazy_property_initializer_writable
    @staticmethod
    def _position_() -> Vec3sT:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _normal_() -> Vec3sT:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _uv_() -> Vec2sT:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _color_() -> Vec4T | Vec4sT:
        return np.ones(4)

    @lazy_property_initializer
    @staticmethod
    def _attributes_o_() -> AttributesBuffer:
        return AttributesBuffer([
            "vec3 a_position",
            "vec3 a_normal",
            "vec2 a_uv",
            "vec4 a_color"
        ])

    @lazy_property
    @staticmethod
    def _attributes_(
        attributes_o: AttributesBuffer,
        position: Vec3sT,
        normal: Vec3sT,
        uv: Vec2sT,
        color: Vec4T | Vec4sT  # TODO
    ) -> AttributesBuffer:
        if len(color.shape) == 1:
            color = color[None].repeat(len(position), axis=0)
        attributes_o.write({
            "a_position": position,
            "a_normal": normal,
            "a_uv": uv,
            "a_color": color
        })
        return attributes_o
