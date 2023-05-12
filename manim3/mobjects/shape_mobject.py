from ..geometries.shape_geometry import ShapeGeometry
from ..lazy.lazy import Lazy
from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.stroke_mobject import StrokeMobject
from ..shape.shape import Shape


class ShapeMobject(MeshMobject):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            self.set_shape(shape)
        self.set_material(enable_phong_lighting=False)

    @Lazy.variable
    @classmethod
    def _shape_(cls) -> Shape:
        return Shape()

    @Lazy.property
    @classmethod
    def _geometry_(
        cls,
        _shape_: Shape
    ) -> ShapeGeometry:
        return ShapeGeometry(_shape_)

    def get_shape(self) -> Shape:
        return self._shape_

    def set_shape(
        self,
        shape: Shape
    ):
        self._shape_ = shape
        return self

    def concatenate(self) -> "ShapeMobject":
        self._shape_ = Shape.concatenate(
            child._shape_
            for child in self.iter_children_by_type(mobject_type=ShapeMobject)
        )
        self.clear()
        return self

    def build_stroke(self) -> StrokeMobject:
        stroke = StrokeMobject()
        stroke._model_matrix_ = self._model_matrix_
        stroke._multi_line_string_ = self._shape_._multi_line_string_
        return stroke

    #def add_stroke(
    #    self,
    #    **kwargs
    #):
    #    self.add(self.build_stroke(**kwargs))
    #    return self
