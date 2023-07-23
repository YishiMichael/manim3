from ...graph_mobjects.graphs.graph_interpolate_handler import GraphInterpolateHandler
from ...mobject.operation_handlers.interpolate_handler import InterpolateHandler
from .shape import Shape


class ShapeInterpolateHandler(InterpolateHandler[Shape]):
    __slots__ = ("_graph_interpolate_handler",)

    def __init__(
        self,
        shape_0: Shape,
        shape_1: Shape
    ) -> None:
        super().__init__(shape_0, shape_1)
        self._graph_interpolate_handler: GraphInterpolateHandler = GraphInterpolateHandler(
            shape_0._graph_, shape_1._graph_
        )

    def interpolate(
        self,
        alpha: float
    ) -> Shape:
        return Shape(self._graph_interpolate_handler.interpolate(alpha))
