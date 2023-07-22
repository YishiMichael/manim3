from ...constants.custom_typing import NP_3f8
from .graphs.line_graph import LineGraph
from .graph_mobject import GraphMobject


class Line(GraphMobject):
    __slots__ = ()

    def __init__(
        self,
        start_point: NP_3f8,
        stop_point: NP_3f8
    ) -> None:
        super().__init__(graph=LineGraph(start_point=start_point, stop_point=stop_point))
