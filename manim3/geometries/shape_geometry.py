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
from ..lazy.interface import (
    Lazy,
    LazyMode
)
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

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _shape_(cls) -> Shape:
        return Shape()

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _geometry_data_(
        cls,
        shape__triangulation: tuple[VertexIndexType, Vec2sT]
    ) -> GeometryData:
        index, coords = shape__triangulation
        position = SpaceUtils.increase_dimension(coords)
        normal = SpaceUtils.increase_dimension(np.zeros_like(coords), z_value=1.0)
        return GeometryData(
            index=index,
            position=position,
            normal=normal,
            uv=coords
        )
