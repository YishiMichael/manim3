__all__ = ["Geometry"]


import numpy as np

from ..custom_typing import (
    Vector2ArrayType,
    Vector3ArrayType,
    VertexIndicesType
)
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)
from ..utils.renderable import (
    AttributeBuffer,
    IndexBuffer,
    Renderable
)


class Geometry(Renderable):
    def __init__(
        self,
        index: VertexIndicesType,
        position: Vector3ArrayType,
        uv: Vector2ArrayType
    ):
        super().__init__()
        self._index_ = index
        self._position_ = position
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
    def _position_() -> Vector3ArrayType:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _a_position_o_() -> AttributeBuffer:
        return AttributeBuffer("vec3", "v")

    @lazy_property
    @staticmethod
    def _a_position_(
        a_position_o: AttributeBuffer,
        position: Vector3ArrayType
    ) -> AttributeBuffer:
        a_position_o.write(position)
        return a_position_o

    @lazy_property_initializer_writable
    @staticmethod
    def _uv_() -> Vector2ArrayType:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _a_uv_o_() -> AttributeBuffer:
        return AttributeBuffer("vec2", "v")

    @lazy_property
    @staticmethod
    def _a_uv_(
        a_uv_o: AttributeBuffer,
        uv: Vector2ArrayType
    ) -> AttributeBuffer:
        a_uv_o.write(uv)
        return a_uv_o
