import numpy as np

from ....utils.space import SpaceUtils
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
        indices = graph._indices_
        knots = graph._knots_
        n_positions = len(positions)
        length = knots[-1]

        interpolated_indices, residues = Graph._interpolate_knots(knots, np.array((alpha_0, alpha_1)) * length)
        extended_positions = np.concatenate((
            positions,
            SpaceUtils.lerp(
                positions[indices[interpolated_indices, 0]],
                positions[indices[interpolated_indices, 1]],
                residues[:, None]
            )
            #np.array((
            #    SpaceUtils.lerp(
            #        positions[indices[2 * interpolate_index_0]], positions[indices[2 * interpolate_index_0 + 1]]
            #    )(residue_0),
            #    SpaceUtils.lerp(
            #        positions[indices[2 * interpolate_index_1]], positions[indices[2 * interpolate_index_1 + 1]]
            #    )(residue_1)
            #))
        ))
        #interpolated_index_0, residue_0 = cls._interpolate_knots(knots, alpha_0 * length, side="right")
        #interpolated_index_1, residue_1 = cls._interpolate_knots(knots, alpha_1 * length, side="left")
        return Graph(
            positions=extended_positions,
            indices=np.column_stack((
                np.insert(indices[interpolated_indices[0] + 1 : interpolated_indices[1] + 1, 0], 0, n_positions),
                np.append(indices[interpolated_indices[0] : interpolated_indices[1], 1], n_positions + 1)
            ))


            #np.insert(
            #    np.array((n_positions, n_positions + 1)),
            #    1,
            #    indices[interpolated_indices[0] : interpolated_indices[1] + 1].flatten()[1:-1]
            #).reshape((-1, 2))
        )
