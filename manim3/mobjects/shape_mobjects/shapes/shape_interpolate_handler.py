import numpy as np

from ....constants.custom_typing import (
    NP_x2i4,
    NP_x3f8
)
from ....utils.space import SpaceUtils
from ...graph_mobjects.graphs.graph import Graph
from ...mobject.operation_handlers.interpolate_handler import InterpolateHandler
from .shape import Shape


class ShapeInterpolateHandler(InterpolateHandler[Shape]):
    __slots__ = (
        "_interpolated_positions_0",
        "_interpolated_positions_1",
        "_edges"
    )

    def __init__(
        self,
        shape_0: Shape,
        shape_1: Shape
    ) -> None:
        graph_0 = shape_0._graph_
        graph_1 = shape_1._graph_
        positions_0 = graph_0._positions_
        positions_1 = graph_1._positions_
        edges_0 = graph_0._edges_
        edges_1 = graph_1._edges_
        assert len(edges_0)
        assert len(edges_1)

        cumlengths_0 = Graph._get_cumlengths(
            positions=positions_0,
            edges=edges_0
        )
        cumlengths_1 = Graph._get_cumlengths(
            positions=positions_1,
            edges=edges_1
        )
        full_knots_0 = cumlengths_0 * cumlengths_1[-1]
        full_knots_1 = cumlengths_1 * cumlengths_0[-1]
        knots_0 = full_knots_0[1:-1]
        knots_1 = full_knots_1[1:-1]
        outline_edges_0, outline_positions_0, interpolated_indices_0 = Graph._get_decomposed_edges(
            positions=positions_0,
            edges=edges_0,
            insertions=np.arange(len(edges_1) - 1) + len(positions_0),
            full_knots=full_knots_0,
            values=knots_1,
            side="right"
        )
        outline_edges_1, outline_positions_1, interpolated_indices_1 = Graph._get_decomposed_edges(
            positions=positions_1,
            edges=edges_1,
            insertions=np.arange(len(edges_0) - 1) + len(positions_1),
            full_knots=full_knots_1,
            values=knots_0,
            side="left"
        )
        disjoints_0 = Graph._get_disjoints(edges=edges_0)
        disjoints_1 = Graph._get_disjoints(edges=edges_1)
        inlay_edges_0 = Graph._reassemble_edges(
            edges=edges_0,
            transition_indices=disjoints_0,
            prepend=edges_0[0, 0],
            append=edges_0[-1, 1],
            insertion_indices=np.searchsorted(
                disjoints_0,
                interpolated_indices_0[disjoints_1],
                side="right"
            ).astype(np.int32),
            insertions=disjoints_1 + len(positions_0)
        )
        inlay_edges_1 = Graph._reassemble_edges(
            edges=edges_1,
            transition_indices=disjoints_1,
            prepend=edges_1[0, 0],
            append=edges_1[-1, 1],
            insertion_indices=np.searchsorted(
                disjoints_1,
                interpolated_indices_1[disjoints_0],
                side="left"
            ).astype(np.int32),
            insertions=disjoints_0 + len(positions_1)
        )
        interpolated_positions_0, interpolated_positions_1, edges = Graph._get_unique_positions(
            positions_0=np.concatenate((
                positions_0,
                outline_positions_0,
                np.average(positions_0, axis=0, keepdims=True)
            )),
            positions_1=np.concatenate((
                positions_1,
                outline_positions_1,
                np.average(positions_1, axis=0, keepdims=True)
            )),
            edges_0=np.concatenate((
                outline_edges_0,
                inlay_edges_0,
                np.ones_like(inlay_edges_1) * (len(positions_0) + len(edges_1) - 1)
            )),
            edges_1=np.concatenate((
                outline_edges_1,
                np.ones_like(inlay_edges_1) * (len(positions_1) + len(edges_0) - 1),
                inlay_edges_1
            ))
        )

        super().__init__(shape_0, shape_1)
        self._interpolated_positions_0: NP_x3f8 = interpolated_positions_0
        self._interpolated_positions_1: NP_x3f8 = interpolated_positions_1
        self._edges: NP_x2i4 = edges

    def interpolate(
        self,
        alpha: float
    ) -> Shape:
        return Shape(Graph(
            positions=SpaceUtils.lerp(self._interpolated_positions_0, self._interpolated_positions_1, alpha),
            edges=self._edges
        ))
