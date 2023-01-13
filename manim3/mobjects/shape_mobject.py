__all__ = [
    "ShapeMobject",
    "ShapeStrokeMobject"
]


from typing import Callable

import moderngl

from ..custom_typing import (
    ColorType,
    Real,
    Vec4T
)
from ..geometries.shape_geometry import ShapeGeometry
from ..geometries.shape_stroke_geometry import ShapeStrokeGeometry
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
    def _geometry_(shape: Shape) -> ShapeGeometry:
        return ShapeGeometry(shape)

    @lazy_property_initializer_writable
    @staticmethod
    def _enable_only_() -> int:
        return moderngl.BLEND

    def set_local_fill(self, color: ColorType | Callable[..., Vec4T]):
        self._color_ = color
        return self

    def set_fill(
        self,
        color: ColorType | Callable[..., Vec4T],
        *,
        broadcast: bool = True
    ):
        for mobject in self.get_descendants(broadcast=broadcast):
            if not isinstance(mobject, ShapeMobject):
                continue
            mobject.set_local_fill(color=color)
        return self


class ShapeStrokeMobject(ShapeMobject):
    def __init__(
        self,
        shape: Shape,
        width: Real,
        round_end: bool = True,
        single_sided: bool = False
    ):
        super().__init__(shape)
        self._width_ = width
        self._round_end_ = round_end
        self._single_sided_ = single_sided

    @lazy_property_initializer_writable
    @staticmethod
    def _width_() -> Real:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _round_end_() -> bool:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _single_sided_() -> bool:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _geometry_(shape: Shape, width: Real, round_end: bool, single_sided: bool) -> ShapeStrokeGeometry:
        return ShapeStrokeGeometry(shape, width, round_end, single_sided)
