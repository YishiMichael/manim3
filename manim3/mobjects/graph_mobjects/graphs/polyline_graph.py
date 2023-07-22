from ....constants.custom_typing import NP_x3f8
from .graph import Graph


class PolylineGraph(Graph):
    __slots__ = ()

    def __init__(
        self,
        points: NP_x3f8
    ) -> None:
        vertices, edges = type(self).args_from_vertex_batches([points])
        super().__init__(
            vertices=vertices,
            edges=edges
        )
