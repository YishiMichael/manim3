__all__ = ["Geometry"]


#import moderngl
import numpy as np

from ..utils.lazy import lazy_property, lazy_property_initializer, lazy_property_initializer_writable
from ..utils.renderable import AttributeBuffer, IndexBuffer, Renderable
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
        #attributes = self.init_geometrys()
        self._indices_ = indices
        self._positions_ = positions
        self._uvs_ = uvs

    @lazy_property_initializer_writable
    @staticmethod
    def _indices_() -> VertexIndicesType:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _index_buffer_o_() -> IndexBuffer:
        return IndexBuffer()

    @lazy_property
    @staticmethod
    def _index_buffer_(
        index_buffer_o: IndexBuffer,
        indices: VertexIndicesType
    ) -> IndexBuffer:
        index_buffer_o._data_ = indices
        return index_buffer_o

    #@_index_buffer_.releaser
    #@staticmethod
    #def _index_buffer_releaser(index_buffer: IndexBuffer) -> None:
    #    index_buffer.release()

    @lazy_property_initializer_writable
    @staticmethod
    def _positions_() -> Vector3ArrayType:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _a_position_o_() -> AttributeBuffer:
        return AttributeBuffer()

    @lazy_property
    @staticmethod
    def _a_position_(
        a_position_o: AttributeBuffer,
        positions: Vector3ArrayType
    ) -> AttributeBuffer:
        a_position_o._data_ = (positions, np.float32)
        return a_position_o

    #@_positions_buffer_.releaser
    #@staticmethod
    #def _positions_buffer_releaser(positions_buffer: moderngl.Buffer) -> None:
    #    positions_buffer.release()

    #@lazy_property_initializer_writable
    #@staticmethod
    #def _uvs_() -> Vector2ArrayType:
    #    return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _uvs_() -> Vector2ArrayType:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _a_uv_o_() -> AttributeBuffer:
        return AttributeBuffer()

    @lazy_property
    @staticmethod
    def _a_uv_(
        a_uv_o: AttributeBuffer,
        uvs: Vector2ArrayType
    ) -> AttributeBuffer:
        a_uv_o._data_ = (uvs, np.float32)
        return a_uv_o

    #@_uvs_buffer_.releaser
    #@staticmethod
    #def _uvs_buffer_releaser(uvs_buffer: moderngl.Buffer) -> None:
    #    uvs_buffer.release()

    #@abstractmethod
    #def init_geometrys(self) -> GeometryAttributes:
    #    pass
