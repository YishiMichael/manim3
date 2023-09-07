#import numpy as np

#from ...mobject.operation_handlers.split_handler import SplitHandler
#from .graph import Graph


#class GraphSplitHandler(SplitHandler[Graph]):
#    __slots__ = ("_graph",)

#    def __init__(
#        self,
#        graph: Graph
#    ) -> None:
#        super().__init__(graph)
#        self._graph: Graph = graph

#    def split(
#        self,
#        alpha_0: float,
#        alpha_1: float
#    ) -> Graph:
#        graph = self._graph
#        positions = graph._positions_
#        edges = graph._edges_
#        if alpha_0 > alpha_1 or not len(edges):
#            return Graph()

#        cumlengths = Graph._get_cumlengths(
#            positions=positions,
#            edges=edges
#        )
#        n_positions = np.int32(len(positions))

#        values = np.array((alpha_0, alpha_1)) * cumlengths[-1]
#        interpolated_indices = np.searchsorted(cumlengths[1:-1], values)
#        all_positions = np.concatenate((
#            positions,
#            Graph._interpolate_positions(
#                positions=positions,
#                edges=edges,
#                full_knots=cumlengths,
#                values=values,
#                indices=interpolated_indices
#            )
#        ))
#        return Graph(
#            positions=all_positions,
#            edges=Graph._reassemble_edges(
#                edges=edges,
#                transition_indices=np.arange(interpolated_indices[0], interpolated_indices[1]),
#                prepend=n_positions,
#                append=n_positions + 1,
#                insertion_indices=np.zeros((0,), dtype=np.int32),
#                insertions=np.zeros((0,), dtype=np.int32)
#            )
#        )
