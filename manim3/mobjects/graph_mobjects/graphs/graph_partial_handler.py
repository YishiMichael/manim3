import numpy as np

from ...mobject.operation_handlers.partial_handler import PartialHandler
from .graph import Graph


class GraphPartialHandler(PartialHandler[Graph]):
    __slots__ = ("_graph",)

    def __init__(
        self,
        graph: Graph
    ) -> None:
        super().__init__(graph)
        self._graph: Graph = graph

    def partial(
        self,
        alpha_0: float,
        alpha_1: float
    ) -> Graph:
        if alpha_0 > alpha_1:
            return Graph()

        graph = self._graph
        positions = graph._positions_
        edges = graph._edges_
        knots = Graph._get_knots(
            positions=positions,
            edges=edges
        )
        n_positions = len(positions) * np.ones((), dtype=np.int32)

        values = np.array((alpha_0, alpha_1)) * knots[-1]
        interpolated_indices = np.searchsorted(knots[1:-1], values)
        all_positions = np.concatenate((
            positions,
            Graph._interpolate_positions(
                positions=positions,
                edges=edges,
                knots=knots,
                values=values,
                indices=interpolated_indices
            )
        ))
        return Graph(
            positions=all_positions,
            edges=Graph._reassemble_edges(
                edges=edges,
                selected_transitions=np.arange(interpolated_indices[0], interpolated_indices[1]),
                prepend=n_positions,
                append=n_positions + 1,
                insertion_indices=np.zeros((0,), dtype=np.int32),
                insertion_values=np.zeros((0,), dtype=np.int32)
            )
        )
