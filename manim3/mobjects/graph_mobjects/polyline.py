from ...constants.custom_typing import NP_x3f8
from .graphs.polyline_graph import PolylineGraph
from .graph_mobject import GraphMobject


class Polyline(GraphMobject):
    __slots__ = ()

    def __init__(
        self,
        positions: NP_x3f8
    ) -> None:
        super().__init__(PolylineGraph(
            positions=positions
        ))
