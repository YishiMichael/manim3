__all__ = [
    "Cube",
    "Dodecahedron",
    "Icosahedron",
    "Octahedron",
    "Polyhedron",
    "Tetrahedron"
]


import numpy as np

from ..custom_typing import (
    Mat4T,
    Vec2sT,
    Vec3sT
)
from ..mobjects.shape_mobject import ShapeMobject
from ..utils.shape import (
    LineString2D,
    MultiLineString2D,
    Shape
)


class Polyhedron(ShapeMobject):
    def __init__(self, vertices: Vec3sT, faces: np.ndarray[tuple[int, int], np.dtype[np.int_]]):
        super().__init__()
        for face in faces:
            matrix, coords = self._convert_coplanar_vertices(vertices[face])
            # Append the last point to form a closed ring
            ring_coords = np.append(coords, coords[0, None], axis=0)
            shape = ShapeMobject(Shape(MultiLineString2D([LineString2D(ring_coords)])))
            shape._set_model_matrix(matrix)
            self.add(shape)

    @classmethod
    def _convert_coplanar_vertices(cls, vertices: Vec3sT) -> tuple[Mat4T, Vec2sT]:
        assert len(vertices) >= 3
        # We first choose three points that define the plane.
        # Instead of choosing `vertices[:3]`, we choose `vertices[:2]` and the geometric centroid,
        # in order to reduce the chance that they happen to be colinear.
        origin = vertices[0]
        x_axis = vertices[1] - vertices[0]
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, np.average(vertices, axis=0) - origin)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation_matrix = np.vstack((x_axis, y_axis, z_axis))

        transformed = (vertices - origin) @ np.linalg.inv(rotation_matrix)
        assert np.isclose(transformed[:, 2], 0.0).all(), "Vertices are not coplanar"

        matrix = np.identity(4)
        matrix[:3, :3] = rotation_matrix
        matrix[3, :3] = origin
        return matrix, transformed[:, :2]


# Ported from manim community
# /manim/mobject/three_d/polyhedra.py
class Tetrahedron(Polyhedron):
    def __init__(self):
        unit = np.sqrt(2.0) / 4.0
        super().__init__(
            vertices=np.array((
                (unit, unit, unit),
                (unit, -unit, -unit),
                (-unit, unit, -unit),
                (-unit, -unit, unit)
            )),
            faces=np.array((
                (0, 1, 2),
                (3, 0, 2),
                (0, 1, 3),
                (3, 1, 2)
            ))
        )


class Cube(Polyhedron):
    def __init__(self):
        super().__init__(
            vertices=np.array((
                (1.0, 1.0, 1.0),
                (1.0, 1.0, -1.0),
                (1.0, -1.0, 1.0),
                (1.0, -1.0, -1.0),
                (-1.0, 1.0, 1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (-1.0, -1.0, -1.0),
            )),
            faces=np.array((
                (0, 1, 3, 2),
                (4, 5, 7, 6),
                (0, 1, 5, 4),
                (2, 3, 7, 6),
                (0, 2, 6, 4),
                (1, 3, 7, 5)
            ))
        )


class Octahedron(Polyhedron):
    def __init__(self):
        unit = np.sqrt(2.0) / 2.0
        super().__init__(
            vertices=np.array((
                (unit, 0.0, 0.0),
                (-unit, 0.0, 0.0),
                (0.0, unit, 0.0),
                (0.0, -unit, 0.0),
                (0.0, 0.0, unit),
                (0.0, 0.0, -unit)
            )),
            faces=np.array((
                (2, 4, 1),
                (0, 4, 2),
                (4, 3, 0),
                (1, 3, 4),
                (3, 5, 0),
                (1, 5, 3),
                (2, 5, 1),
                (0, 5, 2)
            ))
        )


class Icosahedron(Polyhedron):
    def __init__(self):
        unit_a = (1.0 + np.sqrt(5.0)) / 4.0
        unit_b = 1.0 / 2.0
        super().__init__(
            vertices=np.array((
                (0.0, unit_b, unit_a),
                (0.0, -unit_b, unit_a),
                (0.0, unit_b, -unit_a),
                (0.0, -unit_b, -unit_a),
                (unit_b, unit_a, 0.0),
                (unit_b, -unit_a, 0.0),
                (-unit_b, unit_a, 0.0),
                (-unit_b, -unit_a, 0.0),
                (unit_a, 0.0, unit_b),
                (unit_a, 0.0, -unit_b),
                (-unit_a, 0.0, unit_b),
                (-unit_a, 0.0, -unit_b)
            )),
            faces=np.array((
                (1, 8, 0),
                (1, 5, 7),
                (8, 5, 1),
                (7, 3, 5),
                (5, 9, 3),
                (8, 9, 5),
                (3, 2, 9),
                (9, 4, 2),
                (8, 4, 9),
                (0, 4, 8),
                (6, 4, 0),
                (6, 2, 4),
                (11, 2, 6),
                (3, 11, 2),
                (0, 6, 10),
                (10, 1, 0),
                (10, 7, 1),
                (11, 7, 3),
                (10, 11, 7),
                (10, 11, 6)
            ))
        )


class Dodecahedron(Polyhedron):
    def __init__(self):
        unit_a = (1.0 + np.sqrt(5.0)) / 4.0
        unit_b = (3.0 + np.sqrt(5.0)) / 4.0
        unit_c = 1.0 / 2.0
        super().__init__(
            vertices=np.array((
                (unit_a, unit_a, unit_a),
                (unit_a, unit_a, -unit_a),
                (unit_a, -unit_a, unit_a),
                (unit_a, -unit_a, -unit_a),
                (-unit_a, unit_a, unit_a),
                (-unit_a, unit_a, -unit_a),
                (-unit_a, -unit_a, unit_a),
                (-unit_a, -unit_a, -unit_a),
                (0.0, unit_c, unit_b),
                (0.0, unit_c, -unit_b),
                (0.0, -unit_c, -unit_b),
                (0.0, -unit_c, unit_b),
                (unit_c, unit_b, 0.0),
                (-unit_c, unit_b, 0.0),
                (unit_c, -unit_b, 0.0),
                (-unit_c, -unit_b, 0.0),
                (unit_b, 0.0, unit_c),
                (-unit_b, 0.0, unit_c),
                (unit_b, 0.0, -unit_c),
                (-unit_b, 0.0, -unit_c)
            )),
            faces=np.array((
                (18, 16, 0, 12, 1),
                (3, 18, 16, 2, 14),
                (3, 10, 9, 1, 18),
                (1, 9, 5, 13, 12),
                (0, 8, 4, 13, 12),
                (2, 16, 0, 8, 11),
                (4, 17, 6, 11, 8),
                (17, 19, 5, 13, 4),
                (19, 7, 15, 6, 17),
                (6, 15, 14, 2, 11),
                (19, 5, 9, 10, 7),
                (7, 10, 3, 14, 15)
            ))
        )
