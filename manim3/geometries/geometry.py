__all__ = ["Geometry"]


import moderngl

from ..utils.lazy import lazy_property, lazy_property_initializer_writable
from ..utils.renderable import Renderable
from ..custom_typing import *


#@dataclass
#class GeometryAttributes:
#    indices: VertexIndicesType
#    positions: Vector3ArrayType
#    uvs: Vector2ArrayType


class Geometry(Renderable):
    def __init__(
        self,
        indices: VertexIndicesType,
        positions: Vector3ArrayType,
        uvs: Vector2ArrayType
    ):
        super().__init__()
        #attributes = self.init_geometry_attributes()
        self._indices_ = indices
        self._positions_ = positions
        self._uvs_ = uvs

    @lazy_property_initializer_writable
    @classmethod
    def _indices_(cls) -> VertexIndicesType:
        return NotImplemented

    @lazy_property_initializer_writable
    @classmethod
    def _positions_(cls) -> Vector3ArrayType:
        return NotImplemented

    @lazy_property_initializer_writable
    @classmethod
    def _uvs_(cls) -> Vector2ArrayType:
        return NotImplemented

    @lazy_property
    @classmethod
    def _indices_buffer_(cls, indices: VertexIndicesType) -> moderngl.Buffer:
        return cls._make_buffer(indices)

    @_indices_buffer_.releaser
    @staticmethod
    def _indices_buffer_releaser(indices_buffer: moderngl.Buffer) -> None:
        indices_buffer.release()

    @lazy_property
    @classmethod
    def _positions_buffer_(cls, positions: Vector3ArrayType) -> moderngl.Buffer:
        return cls._make_buffer(positions)

    @_positions_buffer_.releaser
    @staticmethod
    def _positions_buffer_releaser(positions_buffer: moderngl.Buffer) -> None:
        positions_buffer.release()

    @lazy_property
    @classmethod
    def _uvs_buffer_(cls, uvs: Vector2ArrayType) -> moderngl.Buffer:
        return cls._make_buffer(uvs)

    @_uvs_buffer_.releaser
    @staticmethod
    def _uvs_buffer_releaser(uvs_buffer: moderngl.Buffer) -> None:
        uvs_buffer.release()

    #@abstractmethod
    #def init_geometry_attributes(self) -> GeometryAttributes:
    #    pass
