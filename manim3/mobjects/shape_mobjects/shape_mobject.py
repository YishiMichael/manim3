from __future__ import annotations


from typing import (
    Self,
    Unpack
)

import numpy as np

from ...animatables.animatable.animatable import AnimatableMeta
from ...animatables.geometries.mesh import Mesh
from ...animatables.geometries.shape import Shape
from ...animatables.models.model import SetKwargs
from ...constants.custom_typing import (
    NP_x2f8,
    NP_x3i4
)
from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
from ..graph_mobjects.graph_mobject import GraphMobject
from ..mesh_mobjects.mesh_mobject import MeshMobject
#from ..mesh_mobjects.meshes.shape_mesh import ShapeMesh


class ShapeMobject(MeshMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        shape: Shape | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            self._shape_ = shape

    @AnimatableMeta.register_descriptor()
    @AnimatableMeta.register_converter()
    @Lazy.volatile()
    @staticmethod
    def _shape_() -> Shape:
        return Shape()

    @Lazy.property()
    @staticmethod
    def _mesh_(
        shape__triangulation: tuple[NP_x3i4, NP_x2f8]
    ) -> Mesh:
        faces, positions_2d = shape__triangulation
        positions = SpaceUtils.increase_dimension(positions_2d)
        normals = SpaceUtils.increase_dimension(np.zeros_like(positions_2d), z_value=1.0)
        return Mesh(
            positions=positions,
            normals=normals,
            uvs=positions_2d,
            faces=faces
        )

    def build_stroke(
        self: Self,
        **kwargs: Unpack[SetKwargs]
    ) -> GraphMobject:
        stroke = GraphMobject()
        stroke._model_matrix_ = self._model_matrix_.copy()
        stroke._graph_ = self._shape_._graph_.copy()
        stroke.set(**kwargs)
        return stroke

    def add_strokes(
        self: Self,
        **kwargs: Unpack[SetKwargs]
    ) -> Self:
        for mobject in self.iter_descendants():
            if isinstance(mobject, ShapeMobject):
                mobject.add(mobject.build_stroke(**kwargs))
        return self
