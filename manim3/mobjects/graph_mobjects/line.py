from ...constants.custom_typing import NP_3f8
from .graphs.line_graph import LineGraph
from .graph_mobject import GraphMobject


class Line(GraphMobject):
    __slots__ = ()

    def __init__(
        self,
        position_0: NP_3f8,
        position_1: NP_3f8
    ) -> None:
        super().__init__(LineGraph(
            position_0=position_0,
            position_1=position_1
        ))
