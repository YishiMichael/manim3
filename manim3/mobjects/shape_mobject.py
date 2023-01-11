__all__ = ["ShapeMobject"]


import moderngl

from ..geometries.geometry import Geometry
from ..geometries.shape_geometry import ShapeGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer_writable
)
from ..utils.shape import Shape


class ShapeMobject(MeshMobject):
    def __init__(self, shape: Shape):
        super().__init__()
        self._shape_ = shape

    @lazy_property_initializer_writable
    @staticmethod
    def _shape_() -> Shape:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _geometry_(shape: Shape) -> Geometry:
        return ShapeGeometry(shape)

    @lazy_property_initializer_writable
    @staticmethod
    def _enable_only_() -> int:
        return moderngl.BLEND
