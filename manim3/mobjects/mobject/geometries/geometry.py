import numpy as np

from ....constants.custom_typing import (
    NP_x2f8,
    NP_x3f8,
    NP_xu4
)
from ....lazy.lazy import (
    Lazy,
    LazyObject
)
from ....rendering.buffers.attributes_buffer import AttributesBuffer
from ....rendering.buffers.index_buffer import IndexBuffer
from ....rendering.indexed_attributes_buffer import IndexedAttributesBuffer
from ....rendering.mgl_enums import PrimitiveMode


class Geometry(LazyObject):
    __slots__ = ()

    @Lazy.variable_array
    @classmethod
    def _index_(cls) -> NP_xu4:
        return np.zeros((0,), dtype=np.uint32)

    @Lazy.variable_array
    @classmethod
    def _position_(cls) -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.variable_array
    @classmethod
    def _normal_(cls) -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.variable_array
    @classmethod
    def _uv_(cls) -> NP_x2f8:
        return np.zeros((0, 2))

    @Lazy.property
    @classmethod
    def _indexed_attributes_buffer_(
        cls,
        position: NP_x3f8,
        normal: NP_x3f8,
        uv: NP_x2f8,
        index: NP_xu4
    ) -> IndexedAttributesBuffer:
        return IndexedAttributesBuffer(
            attributes_buffer=AttributesBuffer(
                fields=[
                    "vec3 in_position",
                    "vec3 in_normal",
                    "vec2 in_uv"
                ],
                num_vertex=len(position),
                data={
                    "in_position": position,
                    "in_normal": normal,
                    "in_uv": uv
                }
            ),
            index_buffer=IndexBuffer(
                data=index
            ),
            mode=PrimitiveMode.TRIANGLES
        )
