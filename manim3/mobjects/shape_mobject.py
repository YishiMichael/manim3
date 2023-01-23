__all__ = ["ShapeMobject"]


from typing import Callable

from ..custom_typing import (
    ColorType,
    Mat4T,
    Real,
    Vec4T
)
from ..geometries.shape_geometry import ShapeGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.stroke_mobject import StrokeMobject
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)
from ..utils.shape import Shape


class ShapeMobject(MeshMobject):
    def __init__(self, shape: Shape | None = None):
        super().__init__()
        if shape is not None:
            self._set_shape(shape)

    @lazy_property_initializer_writable
    @staticmethod
    def _shape_() -> Shape:
        return Shape()

    @lazy_property
    @staticmethod
    def _geometry_(shape: Shape) -> ShapeGeometry:
        return ShapeGeometry(shape)

    @lazy_property_initializer
    @staticmethod
    def _stroke_mobjects_() -> list[StrokeMobject]:
        return []

    def _set_shape(self, shape: Shape):
        self._shape_ = shape
        for stroke in self._stroke_mobjects_:
            stroke._multi_line_string_ = shape._multi_line_string_3d_
        return self

    def _set_model_matrix(self, matrix: Mat4T):
        super()._set_model_matrix(matrix)
        for stroke in self._stroke_mobjects_:
            stroke._set_model_matrix(matrix)
        return self

    def _set_fill_locally(self, color: ColorType | Callable[..., Vec4T]):
        self._color_ = color
        return self

    def set_fill(
        self,
        color: ColorType | Callable[..., Vec4T],
        *,
        broadcast: bool = True
    ):
        for mobject in self.get_descendants(broadcast=broadcast):
            if not isinstance(mobject, ShapeMobject):
                continue
            mobject._set_fill_locally(color=color)
        return self

    @_stroke_mobjects_.updater
    def _add_stroke_locally(
        self,
        *,
        width: Real | None = None,
        color: ColorType | None = None,
        dilate: Real | None = None,
        single_sided: bool | None = None
    ):
        stroke = StrokeMobject(self._shape_._multi_line_string_3d_)
        if width is not None:
            stroke._width_ = width
        if color is not None:
            stroke._color_ = color
        if dilate is not None:
            stroke._dilate_ = dilate
            stroke._apply_oit_ = True  # TODO
        if single_sided is not None:
            stroke._single_sided_ = single_sided
        stroke._set_model_matrix(self._model_matrix_)
        self._stroke_mobjects_.append(stroke)
        self.add(stroke)
        return self

    def add_stroke(
        self,
        *,
        width: Real | None = None,
        color: ColorType | None = None,
        dilate: Real | None = None,
        single_sided: bool | None = None,
        broadcast: bool = True
    ):
        for mobject in self.get_descendants(broadcast=broadcast):
            if not isinstance(mobject, ShapeMobject):
                continue
            mobject._add_stroke_locally(
                width=width,
                color=color,
                dilate=dilate,
                single_sided=single_sided
            )
        return self
