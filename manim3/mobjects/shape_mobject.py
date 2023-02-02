__all__ = ["ShapeMobject"]


import shapely.geometry
import svgelements as se

from ..custom_typing import (
    ColorType,
    Real
)
from ..geometries.shape_geometry import ShapeGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.stroke_mobject import StrokeMobject
from ..utils.lazy import (
    lazy_property,
    lazy_property_writable
)
from ..utils.shape import Shape


class ShapeMobject(MeshMobject):
    def __init__(self, shape: Shape | shapely.geometry.base.BaseGeometry | se.Shape | None = None):
        self._stroke_mobjects: list[StrokeMobject] = []
        super().__init__()
        if shape is not None:
            if isinstance(shape, Shape):
                shape_obj = shape
            else:
                shape_obj = Shape(shape)
            self.set_shape(shape_obj)
        self.set_style(apply_phong_lighting=False)

    @lazy_property_writable
    @staticmethod
    def _shape_() -> Shape:
        return Shape()

    @lazy_property
    @staticmethod
    def _geometry_o_() -> ShapeGeometry:
        return ShapeGeometry()

    @lazy_property
    @staticmethod
    def _geometry_(
        geometry_o: ShapeGeometry,
        shape: Shape
    ) -> ShapeGeometry:
        geometry_o._shape_ = shape
        return geometry_o

    #@lazy_property_writable
    #@staticmethod
    #def _apply_phong_lighting_() -> bool:
    #    return False

    def set_shape(self, shape: Shape):
        self._shape_ = shape
        for stroke in self._stroke_mobjects:
            stroke._multi_line_string_ = shape._multi_line_string_3d_
        return self

    def _set_fill_locally(
        self,
        *,
        color: ColorType | None = None,
        opacity: Real | None = None,
        apply_oit: bool | None = None,
        ambient_strength: Real | None = None,
        specular_strength: Real | None = None,
        shininess: Real | None = None,
        apply_phong_lighting: bool | None = None
    ):
        super()._set_style_locally(
            color=color,
            opacity=opacity,
            apply_oit=apply_oit,
            ambient_strength=ambient_strength,
            specular_strength=specular_strength,
            shininess=shininess,
            apply_phong_lighting=apply_phong_lighting
        )

    def _add_stroke_locally(
        self,
        *,
        width: Real | None = None,
        single_sided: bool | None = None,
        has_linecap: bool | None = None,
        color: ColorType | None = None,
        opacity: Real | None = None,
        dilate: Real | None = None,
        apply_oit: bool | None = None
    ):
        stroke = StrokeMobject(self._shape_._multi_line_string_3d_)
        stroke.apply_transform(self._model_matrix_)
        stroke._set_style_locally(
            width=width,
            single_sided=single_sided,
            has_linecap=has_linecap,
            color=color,
            opacity=opacity,
            dilate=dilate,
            apply_oit=apply_oit
        )
        self._stroke_mobjects.append(stroke)
        self.add(stroke)
        return self

    def _set_stroke_locally(
        self,
        *,
        index: int | None = None,
        width: Real | None = None,
        single_sided: bool | None = None,
        has_linecap: bool | None = None,
        color: ColorType | None = None,
        opacity: Real | None = None,
        dilate: Real | None = None,
        apply_oit: bool | None = None
    ):
        if self._stroke_mobjects:
            if index is None:
                index = 0
            self._stroke_mobjects[index]._set_style_locally(
                width=width,
                single_sided=single_sided,
                has_linecap=has_linecap,
                color=color,
                opacity=opacity,
                dilate=dilate,
                apply_oit=apply_oit
            )
        else:
            if index is not None:
                raise IndexError
            if any(param is not None for param in (
                width,
                single_sided,
                has_linecap,
                color,
                opacity,
                dilate,
                apply_oit
            )):
                self._add_stroke_locally(
                    width=width,
                    single_sided=single_sided,
                    has_linecap=has_linecap,
                    color=color,
                    opacity=opacity,
                    dilate=dilate,
                    apply_oit=apply_oit
                )
        return self

    def _set_style_locally(
        self,
        *,
        color: ColorType | None = None,
        opacity: Real | None = None,
        apply_oit: bool | None = None,
        ambient_strength: Real | None = None,
        specular_strength: Real | None = None,
        shininess: Real | None = None,
        apply_phong_lighting: bool | None = None,
        stroke_width: Real | None = None,
        stroke_single_sided: bool | None = None,
        stroke_has_linecap: bool | None = None,
        stroke_color: ColorType | None = None,
        stroke_opacity: Real | None = None,
        stroke_dilate: Real | None = None,
        stroke_apply_oit: bool | None = None
    ):
        self._set_fill_locally(
            color=color,
            opacity=opacity,
            apply_oit=apply_oit,
            ambient_strength=ambient_strength,
            specular_strength=specular_strength,
            shininess=shininess,
            apply_phong_lighting=apply_phong_lighting
        )
        self._set_stroke_locally(
            index=None,
            width=stroke_width,
            single_sided=stroke_single_sided,
            has_linecap=stroke_has_linecap,
            color=stroke_color,
            opacity=stroke_opacity,
            dilate=stroke_dilate,
            apply_oit=stroke_apply_oit
        )
        return self

    def set_fill(
        self,
        *,
        color: ColorType | None = None,
        opacity: Real | None = None,
        apply_oit: bool | None = None,
        ambient_strength: Real | None = None,
        specular_strength: Real | None = None,
        shininess: Real | None = None,
        apply_phong_lighting: bool | None = None,
        broadcast: bool = True
    ):
        for mobject in self.iter_descendants(broadcast=broadcast):
            if not isinstance(mobject, ShapeMobject):
                continue
            mobject._set_fill_locally(
                color=color,
                opacity=opacity,
                apply_oit=apply_oit,
                ambient_strength=ambient_strength,
                specular_strength=specular_strength,
                shininess=shininess,
                apply_phong_lighting=apply_phong_lighting
            )
        return self

    def add_stroke(
        self,
        *,
        width: Real | None = None,
        single_sided: bool | None = None,
        has_linecap: bool | None = None,
        color: ColorType | None = None,
        opacity: Real | None = None,
        dilate: Real | None = None,
        apply_oit: bool | None = None,
        broadcast: bool = True
    ):
        for mobject in self.iter_descendants(broadcast=broadcast):
            if not isinstance(mobject, ShapeMobject):
                continue
            mobject._add_stroke_locally(
                width=width,
                single_sided=single_sided,
                has_linecap=has_linecap,
                color=color,
                opacity=opacity,
                dilate=dilate,
                apply_oit=apply_oit
            )
        return self

    def set_stroke(
        self,
        *,
        index: int | None = None,
        width: Real | None = None,
        single_sided: bool | None = None,
        has_linecap: bool | None = None,
        color: ColorType | None = None,
        opacity: Real | None = None,
        dilate: Real | None = None,
        apply_oit: bool | None = None,
        broadcast: bool = True
    ):
        for mobject in self.iter_descendants(broadcast=broadcast):
            if not isinstance(mobject, ShapeMobject):
                continue
            mobject._set_stroke_locally(
                index=index,
                width=width,
                single_sided=single_sided,
                has_linecap=has_linecap,
                color=color,
                opacity=opacity,
                dilate=dilate,
                apply_oit=apply_oit
            )
        return self

    def set_style(
        self,
        *,
        color: ColorType | None = None,
        opacity: Real | None = None,
        apply_oit: bool | None = None,
        ambient_strength: Real | None = None,
        specular_strength: Real | None = None,
        shininess: Real | None = None,
        apply_phong_lighting: bool | None = None,
        stroke_width: Real | None = None,
        stroke_single_sided: bool | None = None,
        stroke_has_linecap: bool | None = None,
        stroke_color: ColorType | None = None,
        stroke_opacity: Real | None = None,
        stroke_dilate: Real | None = None,
        stroke_apply_oit: bool | None = None,
        broadcast: bool = True
    ):
        for mobject in self.iter_descendants(broadcast=broadcast):
            if not isinstance(mobject, ShapeMobject):
                continue
            mobject._set_style_locally(
                color=color,
                opacity=opacity,
                apply_oit=apply_oit,
                ambient_strength=ambient_strength,
                specular_strength=specular_strength,
                shininess=shininess,
                apply_phong_lighting=apply_phong_lighting,
                stroke_width=stroke_width,
                stroke_single_sided=stroke_single_sided,
                stroke_has_linecap=stroke_has_linecap,
                stroke_color=stroke_color,
                stroke_opacity=stroke_opacity,
                stroke_dilate=stroke_dilate,
                stroke_apply_oit=stroke_apply_oit
            )
        return self
