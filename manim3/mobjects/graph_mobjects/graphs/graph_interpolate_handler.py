import numpy as np

from ....constants.custom_typing import (
    NP_x2i4,
    NP_x3f8
)
from ....utils.space import SpaceUtils
from ...mobject.operation_handlers.interpolate_handler import InterpolateHandler
from .graph import Graph


class GraphInterpolateHandler(InterpolateHandler[Graph]):
    __slots__ = (
        "_interpolated_positions_0",
        "_interpolated_positions_1",
        "_edges"
    )

    def __init__(
        self,
        graph_0: Graph,
        graph_1: Graph
    ) -> None:
        positions_0 = graph_0._positions_
        positions_1 = graph_1._positions_
        edges_0 = graph_0._edges_
        edges_1 = graph_1._edges_
        assert len(edges_0)
        assert len(edges_1)

        knots_0 = Graph._get_knots(positions_0, edges_0)
        knots_1 = Graph._get_knots(positions_1, edges_1)
        aligned_knots_0 = knots_0 * knots_1[-1]
        aligned_knots_1 = knots_1 * knots_0[-1]
        real_knots_0 = aligned_knots_0[1:-1]
        real_knots_1 = aligned_knots_1[1:-1]
        interpolated_indices_0 = np.searchsorted(
            real_knots_0,
            real_knots_1,
            side="right"
        )
        interpolated_indices_1 = np.searchsorted(
            real_knots_1,
            real_knots_0,
            side="left"
        )
        outline_positions_0 = Graph._interpolate_positions(
            positions=positions_0,
            edges=edges_0,
            knots=aligned_knots_0,
            values=real_knots_1,
            indices=interpolated_indices_0
        )
        outline_positions_1 = Graph._interpolate_positions(
            positions=positions_1,
            edges=edges_1,
            knots=aligned_knots_1,
            values=real_knots_0,
            indices=interpolated_indices_1
        )
        outline_edges_0, outline_edges_1 = Graph._align_edges(
            edges_0=edges_0,
            edges_1=edges_1,
            selected_transitions_0=np.arange(len(edges_0) - 1),
            selected_transitions_1=np.arange(len(edges_1) - 1),
            insertion_indices_0=interpolated_indices_0,
            insertion_indices_1=interpolated_indices_1,
            insertion_indices_offset_0=len(positions_0),
            insertion_indices_offset_1=len(positions_1)
        )
        interpolated_positions_0, interpolated_positions_1, edges = Graph._get_unique_positions(
            positions_0=np.concatenate((
                positions_0,
                outline_positions_0
            )),
            positions_1=np.concatenate((
                positions_1,
                outline_positions_1
            )),
            edges_0=outline_edges_0,
            edges_1=outline_edges_1
        )

        super().__init__(graph_0, graph_1)
        self._interpolated_positions_0: NP_x3f8 = interpolated_positions_0
        self._interpolated_positions_1: NP_x3f8 = interpolated_positions_1
        self._edges: NP_x2i4 = edges

    def interpolate(
        self,
        alpha: float
    ) -> Graph:
        return Graph(
            positions=SpaceUtils.lerp(self._interpolated_positions_0, self._interpolated_positions_1, alpha),
            edges=self._edges
        )
