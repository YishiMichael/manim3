import numpy as np

from ...animatables.geometries.shape import Shape
from ...constants.custom_typing import NP_x2f8
#from .shapes.polygon_shape import PolygonShape
from .shape_mobject import ShapeMobject

#from ...constants.custom_typing import NP_x2f8
#from .shape import Shape


class Polygon(ShapeMobject):
    __slots__ = ()

    def __init__(
        self,
        positions: NP_x2f8
    ) -> None:
        super().__init__(Shape(
            positions=positions,
            counts=np.array((len(positions)),)
        ))
