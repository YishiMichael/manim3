from ...lazy.lazy import Lazy
from ..graph_mobjects.graph_mobject import GraphMobject
from ..mesh_mobjects.mesh_mobject import MeshMobject
from ..mesh_mobjects.meshes.shape_mesh import ShapeMesh
from .shapes.shape import Shape


class ShapeMobject(MeshMobject):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            self._shape_ = shape

    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _shape_() -> Shape:
        return Shape()

    @Lazy.property(hasher=Lazy.branch_hasher)
    @staticmethod
    def _mesh_(
        shape: Shape
    ) -> ShapeMesh:
        return ShapeMesh(shape)

    def build_stroke(self) -> GraphMobject:
        stroke = GraphMobject()
        stroke._model_matrix_ = self._model_matrix_
        stroke._graph_ = self._shape_._graph_
        return stroke
