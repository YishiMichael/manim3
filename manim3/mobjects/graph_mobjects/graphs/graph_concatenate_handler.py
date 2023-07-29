import numpy as np

from ...mobject.operation_handlers.concatenate_handler import ConcatenateHandler
from .graph import Graph


class GraphConcatenateHandler(ConcatenateHandler[Graph]):
    __slots__ = ("_graphs",)

    def __init__(
        self,
        *graphs: Graph
    ) -> None:
        super().__init__(*graphs)
        self._graphs: list[Graph] = list(graphs)

    def concatenate(self) -> Graph:
        graphs = self._graphs
        if not graphs:
            return Graph()

        positions = np.concatenate([
            graph._positions_
            for graph in graphs
        ])
        offsets = np.insert(np.cumsum([
            len(graph._positions_)
            for graph in graphs[:-1]
        ], dtype=np.int32), 0, 0)
        edges = np.concatenate([
            graph._edges_ + offset
            for graph, offset in zip(graphs, offsets, strict=True)
        ])
        return Graph(
            positions=positions,
            edges=edges
        )
