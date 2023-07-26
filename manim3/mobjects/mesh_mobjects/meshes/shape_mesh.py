import numpy as np

from ....utils.space import SpaceUtils
from ...shape_mobjects.shapes.shape import Shape
from .mesh import Mesh


class ShapeMesh(Mesh):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape
    ) -> None:
        faces, positions_2d = shape._triangulation_
        positions = SpaceUtils.increase_dimension(positions_2d)
        normals = SpaceUtils.increase_dimension(np.zeros_like(positions_2d), z_value=1.0)

        super().__init__(
            positions=positions,
            normals=normals,
            uvs=positions_2d,
            faces=faces
        )
