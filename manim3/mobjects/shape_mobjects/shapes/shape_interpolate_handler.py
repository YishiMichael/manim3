import numpy as np

from ....constants.custom_typing import (
    NP_x2i4,
    NP_x3f8
)
from ....utils.space import SpaceUtils
from ...mobject.operation_handlers.interpolate_handler import InterpolateHandler
from ...graph_mobjects.graphs.graph import Graph
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

        knots_0 = Graph._get_knots(
            positions=positions_0,
            edges=edges_0
        )
        knots_1 = Graph._get_knots(
            positions=positions_1,
            edges=edges_1
        )
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
        boundary_positions_0 = Graph._interpolate_positions(
            positions=positions_0,
            edges=edges_0,
            knots=aligned_knots_0,
            values=real_knots_1,
            indices=interpolated_indices_0
        )
        boundary_positions_1 = Graph._interpolate_positions(
            positions=positions_1,
            edges=edges_1,
            knots=aligned_knots_1,
            values=real_knots_0,
            indices=interpolated_indices_1
        )
        boundary_edges_0, boundary_edges_1 = Graph._align_edges(
            edges_0=edges_0,
            edges_1=edges_1,
            selected_transitions_0=np.arange(len(edges_0) - 1),
            selected_transitions_1=np.arange(len(edges_1) - 1),
            insertion_indices_0=interpolated_indices_0,
            insertion_indices_1=interpolated_indices_1,
            insertion_indices_offset_0=len(positions_0),
            insertion_indices_offset_1=len(positions_1)
        )
        disjoints_0 = Graph._get_disjoints(edges=edges_0)
        disjoints_1 = Graph._get_disjoints(edges=edges_1)
        inlay_interpolated_indices_0 = np.searchsorted(
            real_knots_0[disjoints_0],
            real_knots_1[disjoints_1],
            side="right"
        )
        inlay_interpolated_indices_1 = np.searchsorted(
            real_knots_1[disjoints_1],
            real_knots_0[disjoints_0],
            side="left"
        )
        inlay_edges_0, inlay_edges_1 = Graph._align_edges(
            edges_0=edges_0,
            edges_1=edges_1,
            selected_transitions_0=disjoints_0,
            selected_transitions_1=disjoints_1,
            insertion_indices_0=inlay_interpolated_indices_0,
            insertion_indices_1=inlay_interpolated_indices_1,
            insertion_indices_offset_0=len(positions_0),
            insertion_indices_offset_1=len(positions_1)
        )
        interpolated_positions_0, interpolated_positions_1, edges = Graph._get_unique_positions(
            positions_0=np.concatenate((
                positions_0,
                boundary_positions_0,
                np.average(positions_0, axis=0, keepdims=True)
            )),
            positions_1=np.concatenate((
                positions_1,
                boundary_positions_1,
                np.average(positions_1, axis=0, keepdims=True)
            )),
            edges_0=np.concatenate((
                boundary_edges_0,
                inlay_edges_0,
                np.ones_like(inlay_edges_1) * (len(positions_0) + len(edges_1) - 1)
            )),
            edges_1=np.concatenate((
                boundary_edges_1,
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
