__all__ = ["Geometry"]


from ..custom_typing import (
    Vec2sT,
    Vec3sT,
    VertexIndexType
)
from ..utils.lazy import (
    LazyBase,
    lazy_property_writable
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
