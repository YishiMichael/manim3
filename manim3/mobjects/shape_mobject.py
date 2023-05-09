from ..custom_typing import ColorT
from ..geometries.shape_geometry import ShapeGeometry
from ..lazy.interface import Lazy
from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.stroke_mobject import StrokeMobject
from ..utils.shape import Shape


class ShapeMobject(MeshMobject):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            self.set_shape(shape)
        self.set_style(enable_phong_lighting=False)

    @Lazy.variable
    @classmethod
    def _shape_(cls) -> Shape:
        return Shape()

    @Lazy.variable_shared
    @classmethod
    def _enable_phong_lighting_(cls) -> bool:
        return False

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

    def build_stroke(
        self,
        width: float | None = None,
        single_sided: bool | None = None,
        has_linecap: bool | None = None,
        color: ColorT | None = None,
        opacity: float | None = None,
        dilate: float | None = None,
        is_transparent: bool | None = None
    ) -> StrokeMobject:
        stroke = StrokeMobject()
        stroke._model_matrix_ = self._model_matrix_
        stroke._multi_line_string_ = self._shape_._multi_line_string_
        return stroke.set_style(
            width=width,
            single_sided=single_sided,
            has_linecap=has_linecap,
            color=color,
            opacity=opacity,
            dilate=dilate,
            is_transparent=is_transparent
        )
