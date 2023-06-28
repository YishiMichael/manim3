from ..geometries.shape_geometry import ShapeGeometry
from ..lazy.lazy import Lazy
from ..shape.shape import Shape
from .mesh_mobject import MeshMobject
from .mobject import StyleMeta
from .stroke_mobject import StrokeMobject


class ShapeMobject(MeshMobject):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            self._shape_ = shape

    @StyleMeta.register(
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

    def build_stroke(self) -> StrokeMobject:
        stroke = StrokeMobject()
        stroke._model_matrix_ = self._model_matrix_
        stroke._multi_line_string_ = self._shape_._multi_line_string_
        stroke._color_ = self._color_
        stroke._opacity_ = self._opacity_
        stroke._weight_ = self._weight_
        return stroke
