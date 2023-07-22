from ..lazy.lazy import Lazy
from .mesh_mobject import MeshMobject
from .mobject.geometries.shape_geometry import ShapeGeometry
from .mobject.mobject_style_meta import MobjectStyleMeta
from .mobject.shape.shape import Shape
from .graph_mobject import GraphMobject


class ShapeMobject(MeshMobject):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            self._shape_ = shape

    @MobjectStyleMeta.register(
        partial_method=Shape.partial,
        interpolate_method=Shape.interpolate,
        concatenate_method=Shape.concatenate
    )
    @Lazy.variable
    @classmethod
    def _shape_(cls) -> Shape:
        return Shape()

    @Lazy.property
    @classmethod
    def _geometry_(
        cls,
        shape: Shape
    ) -> ShapeGeometry:
        return ShapeGeometry(shape)

    def build_stroke(self) -> GraphMobject:
        stroke = GraphMobject()
        stroke._model_matrix_ = self._model_matrix_
        stroke._graph_ = self._shape_._graph_
        return stroke
