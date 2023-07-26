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
        "_aligned_positions_0",
        "_aligned_positions_1",
        "_indices"
    )

    def __init__(
        self,
        graph_0: Graph,
        graph_1: Graph
    ) -> None:
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
            )
        ))
        extended_positions_1 = np.concatenate((
            positions_1,
            SpaceUtils.lerp(
                positions_1[indices_1[interpolated_indices_1, 0]],
                positions_1[indices_1[interpolated_indices_1, 1]],
                residues_1[:, None]
            )
        ))
        extended_indices_0 = np.column_stack((
            np.insert(np.insert(
                indices_0[1:, 0],
                interpolated_indices_0,
                np.arange(len(positions_0), len(extended_positions_0))
            ), 0, indices_0[0, 0]),
            np.append(np.insert(
                indices_0[:-1, 1],
                interpolated_indices_0,
                np.arange(len(positions_0), len(extended_positions_0))
            ), indices_0[-1, 1])
        ))
        extended_indices_1 = np.column_stack((
            np.insert(np.insert(
                indices_1[1:, 0],
                interpolated_indices_1,
                np.arange(len(positions_1), len(extended_positions_1))
            ), 0, indices_1[0, 0]),
            np.append(np.insert(
                indices_1[:-1, 1],
                interpolated_indices_1,
                np.arange(len(positions_1), len(extended_positions_1))
            ), indices_1[-1, 1])
        ))
        aligned_indices, indices_inverse = np.unique(
            np.array((extended_indices_0.flatten(), extended_indices_1.flatten())),
            axis=1,
            return_inverse=True
        )

        super().__init__(graph_0, graph_1)
        self._aligned_positions_0: NP_x3f8 = extended_positions_0[aligned_indices[0]]
        self._aligned_positions_1: NP_x3f8 = extended_positions_1[aligned_indices[1]]
        self._indices: NP_x2i4 = indices_inverse.reshape((-1, 2))

    def interpolate(
        self,
        alpha: float
    ) -> Graph:
        return Graph(
            positions=SpaceUtils.lerp(self._aligned_positions_0, self._aligned_positions_1, alpha),
            indices=self._indices
        )
