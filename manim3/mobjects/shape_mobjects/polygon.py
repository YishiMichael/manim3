from ...constants.custom_typing import NP_x2f8
from .shape_mobject import ShapeMobject
from .shapes.polygon_shape import PolygonShape


class Polygon(ShapeMobject):
    __slots__ = ()

    def __init__(
        self,
        positions: NP_x2f8
    ) -> None:
        super().__init__(PolygonShape(
            positions=positions
        ))
