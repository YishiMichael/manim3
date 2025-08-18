from __future__ import annotations


from typing import Self

import numpy as np

from ...animatables.arrays.model_matrix import ModelMatrix
from ...animatables.shape import Shape
from ...constants.custom_typing import NP_x2f8
from ...constants.custom_typing import NP_x3f8
from .shape_mobject import ShapeMobject


class Polygon(ShapeMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        coordinates: NP_x2f8
    ) -> None:
        super().__init__(Shape(
            coordinates=coordinates,
            counts=np.array((len(coordinates),))
        ))

    @classmethod
    def from_positions(
        cls: type[Self],
        positions: NP_x3f8
    ) -> Self:
        assert len(positions) >= 3
        sample_0 = np.average(positions, axis=0)
        sample_1 = positions[0]
        sample_2 = positions[1]

        x_axis = (x_direction := sample_1 - sample_0) / np.linalg.norm(x_direction)
        z_axis = (z_direction := np.cross(x_axis, sample_2 - sample_0)) / np.linalg.norm(z_direction)
        y_axis = np.cross(z_axis, x_axis)
        matrix = np.identity(4)
        matrix[:-1] = np.column_stack((x_axis, y_axis, z_axis, sample_0))

        transformed = ModelMatrix._apply_multiple(np.linalg.inv(matrix), positions)
        transformed_xy = transformed[:, :2]
        transformed_z = transformed[:, 2]
        assert np.isclose(transformed_z, 0.0).all(), "Positions are not coplanar"
        return cls(transformed_xy).apply(matrix)
