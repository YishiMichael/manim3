from ...constants.custom_typing import NP_3f8
from .graphs.line_graph import LineGraph
from .graph_mobject import GraphMobject


class Line(GraphMobject):
    __slots__ = ()

    def __init__(
        self,
        start_position: NP_3f8,
        stop_position: NP_3f8
    ) -> None:
        super().__init__(LineGraph(
            start_position=start_position,
            stop_position=stop_position
        ))
