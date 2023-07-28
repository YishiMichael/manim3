from ...constants.custom_typing import NP_x2f8
from .shapes.polygon_shape import PolygonShape
from .shape_mobject import ShapeMobject


class Polygon(ShapeMobject):
    __slots__ = ()

    def __init__(
        self,
        positions: NP_x2f8
    ) -> None:
        super().__init__(PolygonShape(
            positions=positions
        ))
