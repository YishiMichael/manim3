from ...lazy.lazy import Lazy
from ..graph_mobjects.graph_mobject import GraphMobject
from ..mesh_mobjects.mesh_mobject import MeshMobject
from ..mesh_mobjects.meshes.shape_mesh import ShapeMesh
#from ..mobject.style_meta import StyleMeta
from .shapes.shape import Shape
#from .shapes.shape_concatenate_handler import ShapeConcatenateHandler
#from .shapes.shape_interpolate_handler import ShapeInterpolateHandler
#from .shapes.shape_split_handler import ShapeSplitHandler


class ShapeMobject(MeshMobject):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            self._shape_ = shape

    #@StyleMeta.register(
    #    split_operation=ShapeSplitHandler,
    #    concatenate_operation=ShapeConcatenateHandler,
    #    interpolate_operation=ShapeInterpolateHandler
    #)
    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _shape_() -> Shape:
        return Shape()

    @Lazy.property()
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
