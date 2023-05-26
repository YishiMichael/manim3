import numpy as np

from ..geometries.geometry import Geometry
from ..shape.shape import Shape
from ..utils.space import SpaceUtils


class ShapeGeometry(Geometry):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape
    ) -> None:
        index, points = shape._triangulation_
        position = SpaceUtils.increase_dimension(points)
        normal = SpaceUtils.increase_dimension(np.zeros_like(points), z_value=1.0)

        super().__init__()
        self._index_ = index
        self._position_ = position
        self._normal_ = normal
        self._uv_ = points
        #return GeometryData(
        #    index=index,
        #    position=position,
        #    normal=normal,
        #    uv=points
        #)
