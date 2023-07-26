from ....constants.custom_typing import NP_x3f8
from .graph import Graph


class PolylineGraph(Graph):
    __slots__ = ()

    def __init__(
        self,
        positions: NP_x3f8
    ) -> None:
        super().__init__(
            positions=positions,
            edges=Graph._get_consecutive_edges(len(positions), is_ring=False)
        )
