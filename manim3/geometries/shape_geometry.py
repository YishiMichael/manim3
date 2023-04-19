__all__ = ["ShapeGeometry"]


import numpy as np

from ..custom_typing import (
    Vec2sT,
    VertexIndexType
)
from ..geometries.geometry import (
    Geometry,
    GeometryData
)
from ..lazy.interface import Lazy
from ..utils.shape import Shape
from ..utils.space import SpaceUtils


class ShapeGeometry(Geometry):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            self._shape_ = shape

    @Lazy.variable
    @classmethod
    def _shape_(cls) -> Shape:
        return Shape()

    @Lazy.property_external
    @classmethod
    def _geometry_data_(
        cls,
        shape__triangulation: tuple[VertexIndexType, Vec2sT]
    ) -> GeometryData:
        index, points = shape__triangulation
        position = SpaceUtils.increase_dimension(points)
        normal = SpaceUtils.increase_dimension(np.zeros_like(points), z_value=1.0)
        return GeometryData(
            index=index,
            position=position,
            normal=normal,
            uv=points
        )
