from ...graph_mobjects.graphs.graph_partial_handler import GraphPartialHandler
from ...mobject.operation_handlers.partial_handler import PartialHandler
from .shape import Shape


class ShapePartialHandler(PartialHandler[Shape]):
    __slots__ = ("_graph_partial_handler",)

    def __init__(
        self,
        shape: Shape
    ) -> None:
        super().__init__(shape)
        self._graph_partial_handler: GraphPartialHandler = GraphPartialHandler(
            shape._graph_
        )

    def partial(
        self,
        alpha_0: float,
        alpha_1: float
    ) -> Shape:
        return Shape(self._graph_partial_handler.partial(alpha_0, alpha_1))
