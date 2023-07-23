from ...graph_mobjects.graphs.graph_concatenate_handler import GraphConcatenateHandler
from ...mobject.operation_handlers.concatenate_handler import ConcatenateHandler
from .shape import Shape


class ShapeConcatenateHandler(ConcatenateHandler[Shape]):
    __slots__ = ("_graph_concatenate_handler",)

    def __init__(
        self,
        *shapes: Shape
    ) -> None:
        super().__init__(*shapes)
        self._graph_concatenate_handler: GraphConcatenateHandler = GraphConcatenateHandler(*(
            shape._graph_ for shape in shapes
        ))

    def concatenate(self) -> Shape:
        return Shape(self._graph_concatenate_handler.concatenate())
