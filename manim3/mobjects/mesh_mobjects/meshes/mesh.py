import numpy as np


from ....constants.custom_typing import (
    NP_x2f8,
    NP_x3f8,
    NP_x3i4
)
from ....lazy.lazy import Lazy
from ....rendering.buffers.attributes_buffer import AttributesBuffer
from ....rendering.buffers.index_buffer import IndexBuffer
from ....rendering.indexed_attributes_buffer import IndexedAttributesBuffer
from ....rendering.mgl_enums import PrimitiveMode
from ...mobject.mobject_attributes.mobject_attribute import MobjectAttribute


class Mesh(MobjectAttribute):
    __slots__ = ()

    def __init__(
        self,
        positions: NP_x3f8 | None = None,
        normals: NP_x3f8 | None = None,
        uvs: NP_x2f8 | None = None,
        faces: NP_x3i4 | None = None
    ) -> None:
        super().__init__()
        if positions is not None:
            self._positions_ = positions
        if normals is not None:
            self._normals_ = normals
        if uvs is not None:
            self._uvs_ = uvs
        if faces is not None:
            self._faces_ = faces

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _positions_() -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _normals_() -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _uvs_() -> NP_x2f8:
        return np.zeros((0, 2))

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _faces_() -> NP_x3i4:
        return np.zeros((0, 3), dtype=np.int32)

    @Lazy.property()
    @staticmethod
    def _indexed_attributes_buffer_(
        positions: NP_x3f8,
        normals: NP_x3f8,
        uvs: NP_x2f8,
        faces: NP_x3i4
    ) -> IndexedAttributesBuffer:
        return IndexedAttributesBuffer(
            attributes_buffer=AttributesBuffer(
                fields=[
                    "vec3 in_position",
                    "vec3 in_normal",
                    "vec2 in_uv"
                ],
                num_vertex=len(positions),
                data={
                    "in_position": positions,
                    "in_normal": normals,
                    "in_uv": uvs
                }
            ),
            index_buffer=IndexBuffer(
                data=faces.flatten()
            ),
            mode=PrimitiveMode.TRIANGLES
        )
