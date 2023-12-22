from __future__ import annotations


import math
from typing import Self

import numpy as np

from ...animatables.arrays.model_matrix import ModelMatrix
from ...constants.custom_typing import (
    NP_x3f8,
    NP_xxi4
)
from .polygon import Polygon
from .shape_mobject import ShapeMobject


class Polyhedron(ShapeMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        positions: NP_x3f8,
        faces: NP_xxi4
    ) -> None:
        super().__init__()
        self.add(*(
            type(self)._get_transformed_face(positions[face])
            for face in faces
        ))

    @classmethod
    def _get_transformed_face(
        cls: type[Self],
        positions: NP_x3f8
    ) -> Polygon:
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
        return Polygon(transformed_xy).apply(matrix)


# The five Platonic solids, ported from manim community `/manim/mobject/three_d/polyhedra.py`
# All these polyhedra have all positions sitting on the unit sphere.
class Tetrahedron(Polyhedron):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        a = 1.0 / math.sqrt(3.0)
        super().__init__(
            positions=np.array((
                ( a,  a,  a),
                ( a, -a, -a),
                (-a,  a, -a),
                (-a, -a,  a)
            )),
            faces=np.array((
                (0, 1, 2),
                (3, 0, 2),
                (1, 0, 3),
                (2, 1, 3)
            ))
        )


class Cube(Polyhedron):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        a = 1.0 / math.sqrt(3.0)
        super().__init__(
            positions=np.array((
                ( a,  a,  a),
                ( a,  a, -a),
                ( a, -a,  a),
                ( a, -a, -a),
                (-a,  a,  a),
                (-a,  a, -a),
                (-a, -a,  a),
                (-a, -a, -a)
            )),
            faces=np.array((
                (0, 2, 3, 1),
                (4, 5, 7, 6),
                (0, 1, 5, 4),
                (2, 6, 7, 3),
                (0, 4, 6, 2),
                (1, 3, 7, 5)
            ))
        )


class Octahedron(Polyhedron):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        a = 1.0
        super().__init__(
            positions=np.array((
                (  a, 0.0, 0.0),
                ( -a, 0.0, 0.0),
                (0.0,   a, 0.0),
                (0.0,  -a, 0.0),
                (0.0, 0.0,   a),
                (0.0, 0.0,  -a)
            )),
            faces=np.array((
                (0, 2, 4),
                (2, 1, 4),
                (1, 3, 4),
                (3, 0, 4),
                (0, 3, 5),
                (3, 1, 5),
                (1, 2, 5),
                (2, 0, 5)
            ))
        )


class Dodecahedron(Polyhedron):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        a = (math.sqrt(5.0) + 1.0) / (2.0 * math.sqrt(3.0))
        b = 1.0 / math.sqrt(3.0)
        c = (math.sqrt(5.0) - 1.0) / (2.0 * math.sqrt(3.0))
        super().__init__(
            positions=np.array((
                (  b,   b,   b),
                (  b,   b,  -b),
                (  b,  -b,   b),
                (  b,  -b,  -b),
                ( -b,   b,   b),
                ( -b,   b,  -b),
                ( -b,  -b,   b),
                ( -b,  -b,  -b),
                (0.0,   a,   c),
                (0.0,   a,  -c),
                (0.0,  -a,   c),
                (0.0,  -a,  -c),
                (  c, 0.0,   a),
                ( -c, 0.0,   a),
                (  c, 0.0,  -a),
                ( -c, 0.0,  -a),
                (  a,   c, 0.0),
                (  a,  -c, 0.0),
                ( -a,   c, 0.0),
                ( -a,  -c, 0.0)
            )),
            faces=np.array((
                ( 8,  0, 16,  1,  9),
                ( 9,  5, 18,  4,  8),
                (10,  6, 19,  7, 11),
                (11,  3, 17,  2, 10),
                (12,  0,  8,  4, 13),
                (13,  6, 10,  2, 12),
                (14,  3, 11,  7, 15),
                (15,  5,  9,  1, 14),
                (16,  0, 12,  2, 17),
                (17,  3, 14,  1, 16),
                (18,  5, 15,  7, 19),
                (19,  6, 13,  4, 18)
            ))
        )


class Icosahedron(Polyhedron):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        a = math.sqrt((1.0 + 1.0 / math.sqrt(5.0)) / 2.0)
        b = math.sqrt((1.0 - 1.0 / math.sqrt(5.0)) / 2.0)
        super().__init__(
            positions=np.array((
                (0.0,   a,   b),
                (0.0,   a,  -b),
                (0.0,  -a,   b),
                (0.0,  -a,  -b),
                (  b, 0.0,   a),
                ( -b, 0.0,   a),
                (  b, 0.0,  -a),
                ( -b, 0.0,  -a),
                (  a,   b, 0.0),
                (  a,  -b, 0.0),
                ( -a,   b, 0.0),
                ( -a,  -b, 0.0)
            )),
            faces=np.array((
                ( 0,  8,  1),
                ( 1, 10,  0),
                ( 2, 11,  3),
                ( 3,  9,  2),
                ( 4,  0,  5),
                ( 5,  2,  4),
                ( 6,  3,  7),
                ( 7,  1,  6),
                ( 8,  4,  9),
                ( 9,  6,  8),
                (10,  7, 11),
                (11,  5, 10),
                ( 8,  0,  4),
                ( 0, 10,  5),
                (11,  2,  5),
                ( 2,  9,  4),
                ( 9,  3,  6),
                ( 3, 11,  7),
                (10,  1,  7),
                ( 1,  8,  6)
            ))
        )
