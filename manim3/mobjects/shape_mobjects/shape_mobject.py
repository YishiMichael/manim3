from ...lazy.lazy import Lazy
from ..graph_mobjects.graph_mobject import GraphMobject
from ..mesh_mobjects.mesh_mobject import MeshMobject
from ..mesh_mobjects.meshes.shape_mesh import ShapeMesh
from ..mobject.operation_handlers.mobject_operation import MobjectOperation
from .shapes.shape import Shape
from .shapes.shape_concatenate_handler import ShapeConcatenateHandler
from .shapes.shape_interpolate_handler import ShapeInterpolateHandler
from .shapes.shape_partial_handler import ShapePartialHandler


class ShapeMobject(MeshMobject):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            self._shape_ = shape

    @MobjectOperation.register(
        partial=ShapePartialHandler,
        interpolate=ShapeInterpolateHandler,
        concatenate=ShapeConcatenateHandler
    )
    @Lazy.variable
    @classmethod
    def _shape_(cls) -> Shape:
        return Shape()

    @Lazy.property
    @classmethod
    def _mesh_(
        cls,
        shape: Shape
    ) -> ShapeMesh:
        return ShapeMesh(shape)

    def build_stroke(self) -> GraphMobject:
        stroke = GraphMobject()
        stroke._model_matrix_ = self._model_matrix_
        stroke._graph_ = self._shape_._graph_
        return stroke
