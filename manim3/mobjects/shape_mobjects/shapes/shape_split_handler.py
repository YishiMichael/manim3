from ...graph_mobjects.graphs.graph_split_handler import GraphSplitHandler
from ...mobject.operation_handlers.split_handler import SplitHandler
from .shape import Shape


class ShapeSplitHandler(SplitHandler[Shape]):
    __slots__ = ("_graph_split_handler",)

    def __init__(
        self,
        shape: Shape
    ) -> None:
        super().__init__(shape)
        self._graph_split_handler: GraphSplitHandler = GraphSplitHandler(
            shape._graph_
        )

    def split(
        self,
        alpha_0: float,
        alpha_1: float
    ) -> Shape:
        return Shape(self._graph_split_handler.split(alpha_0, alpha_1))
