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
        "_aligned_positions_0",
        "_aligned_positions_1",
        "_indices"
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
        indices_0 = graph_0._indices_
        indices_1 = graph_1._indices_
        assert len(indices_0)
        assert len(indices_1)

        knots_0 = graph_0._knots_ * graph_1._knots_[-1]
        knots_1 = graph_1._knots_ * graph_0._knots_[-1]
        interpolated_indices_0, residues_0 = Graph._interpolate_knots(knots_0, knots_1[1:-1], side="right")
        interpolated_indices_1, residues_1 = Graph._interpolate_knots(knots_1, knots_0[1:-1], side="left")
        extended_positions_0 = np.concatenate((
            positions_0,
            SpaceUtils.lerp(
                positions_0[indices_0[interpolated_indices_0, 0]],
                positions_0[indices_0[interpolated_indices_0, 1]],
                residues_0[:, None]
            ),
            np.average(positions_0, axis=0, keepdims=True)
        ))
        extended_positions_1 = np.concatenate((
            positions_1,
            SpaceUtils.lerp(
                positions_1[indices_1[interpolated_indices_1, 0]],
                positions_1[indices_1[interpolated_indices_1, 1]],
                residues_1[:, None]
            ),
            np.average(positions_1, axis=0, keepdims=True)
        ))

        disjoints_0 = np.flatnonzero(indices_0[:-1, 1] - indices_0[1:, 0])
        disjoints_1 = np.flatnonzero(indices_1[:-1, 1] - indices_1[1:, 0])
        inlay_interpolated_indices_0 = np.searchsorted(disjoints_0, interpolated_indices_0[disjoints_1], side="right")
        inlay_interpolated_indices_1 = np.searchsorted(disjoints_1, interpolated_indices_1[disjoints_0], side="left")
        inlay_indices_0 = np.column_stack((
            np.insert(np.insert(
                indices_0[disjoints_0 + 1, 0],
                inlay_interpolated_indices_0,
                disjoints_1 + len(positions_0)
            ), 0, indices_0[0, 0]),
            np.append(np.insert(
                indices_0[disjoints_0, 1],
                inlay_interpolated_indices_0,
                disjoints_1 + len(positions_0)
            ), indices_0[-1, 1])
        ))
        inlay_indices_1 = np.column_stack((
            np.insert(np.insert(
                indices_1[disjoints_1 + 1, 0],
                inlay_interpolated_indices_1,
                disjoints_0 + len(positions_1)
            ), 0, indices_1[0, 0]),
            np.append(np.insert(
                indices_1[disjoints_1, 1],
                inlay_interpolated_indices_1,
                disjoints_0 + len(positions_1)
            ), indices_1[-1, 1])
        ))
        extended_indices_0 = np.concatenate((
            np.column_stack((
                np.insert(np.insert(
                    indices_0[1:, 0],
                    interpolated_indices_0,
                    np.arange(len(positions_0), len(positions_0) + len(indices_1) - 1)
                ), 0, indices_0[0, 0]),
                np.append(np.insert(
                    indices_0[:-1, 1],
                    interpolated_indices_0,
                    np.arange(len(positions_0), len(positions_0) + len(indices_1) - 1)
                ), indices_0[-1, 1])
            )),
            inlay_indices_0,
            np.ones_like(inlay_indices_1) * (len(positions_0) + len(indices_1) - 1)
        ))
        extended_indices_1 = np.concatenate((
            np.column_stack((
                np.insert(np.insert(
                    indices_1[1:, 0],
                    interpolated_indices_1,
                    np.arange(len(positions_1), len(positions_1) + len(indices_0) - 1)
                ), 0, indices_1[0, 0]),
                np.append(np.insert(
                    indices_1[:-1, 1],
                    interpolated_indices_1,
                    np.arange(len(positions_1), len(positions_1) + len(indices_0) - 1)
                ), indices_1[-1, 1])
            )),
            np.ones_like(inlay_indices_1) * (len(positions_1) + len(indices_0) - 1),
            inlay_indices_1
        ))
        aligned_indices, indices_inverse = np.unique(
            np.array((extended_indices_0.flatten(), extended_indices_1.flatten())),
            axis=1,
            return_inverse=True
        )

        #unique_inlay_indices_0, inlay_indices_inverse_0 = np.unique(inlay_indices_0, return_inverse=True)
        #unique_inlay_indices_1, inlay_indices_inverse_1 = np.unique(inlay_indices_1, return_inverse=True)

        super().__init__(shape_0, shape_1)
        self._aligned_positions_0: NP_x3f8 = extended_positions_0[aligned_indices[0]]
        self._aligned_positions_1: NP_x3f8 = extended_positions_1[aligned_indices[1]]
        self._indices: NP_x2i4 = indices_inverse.reshape((-1, 2))
        #self._aligned_positions_0: NP_x3f8 = np.concatenate((
        #    extended_positions_0[aligned_indices[0]],
        #    extended_positions_0[unique_inlay_indices_0],
        #    np.repeat(np.average(positions_0, axis=0, keepdims=True), len(unique_inlay_indices_1), axis=0)
        #))
        #self._aligned_positions_1: NP_x3f8 = np.concatenate((
        #    extended_positions_1[aligned_indices[1]],
        #    np.repeat(np.average(positions_1, axis=0, keepdims=True), len(unique_inlay_indices_0), axis=0),
        #    extended_positions_1[unique_inlay_indices_1]
        #))
        #self._indices: NP_x2i4 = np.concatenate((
        #    indices_inverse,
        #    inlay_indices_inverse_0 + aligned_indices.shape[1],
        #    inlay_indices_inverse_1 + aligned_indices.shape[1] + len(unique_inlay_indices_0)
        #)).reshape((-1, 2))

    def interpolate(
        self,
        alpha: float
    ) -> Shape:
        return Shape(Graph(
            positions=SpaceUtils.lerp(self._aligned_positions_0, self._aligned_positions_1, alpha),
            indices=self._indices
        ))
