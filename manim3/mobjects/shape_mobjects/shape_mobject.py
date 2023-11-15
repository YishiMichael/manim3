from __future__ import annotations


from typing import (
    Self,
    Unpack
)

import numpy as np

from ...animatables.animatable.animatable import AnimatableActions
from ...animatables.mesh import Mesh
from ...animatables.shape import (
    Shape,
    Triangulation
)
from ...animatables.model import SetKwargs
from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
from ..graph_mobjects.graph_mobject import GraphMobject
from ..mesh_mobjects.mesh_mobject import MeshMobject


class ShapeMobject(MeshMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        shape: Shape | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            self._shape_ = shape

    @AnimatableActions.interpolate.register_descriptor()
    @AnimatableActions.piecewise.register_descriptor()
    @Lazy.volatile()
    @staticmethod
    def _shape_() -> Shape:
        return Shape()

    @Lazy.property()
    @staticmethod
    def _mesh_(
        shape__triangulation: Triangulation
    ) -> Mesh:
        faces = shape__triangulation.faces
        coordinates = shape__triangulation.coordinates
        positions = SpaceUtils.increase_dimension(coordinates)
        normals = SpaceUtils.increase_dimension(np.zeros_like(coordinates), z_value=1.0)
        return Mesh(
            positions=positions,
            normals=normals,
            uvs=coordinates,
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
