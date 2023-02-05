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
from ..utils.color import ColorUtils
from ..utils.lazy import (
    LazyData,
    lazy_basedata,
    lazy_property
)
from ..utils.shape import Shape


class ShapeMobject(MeshMobject):
    def __new__(cls, shape: Shape | shapely.geometry.base.BaseGeometry | se.Shape | None = None):
        #self._stroke_mobjects: list[StrokeMobject] = []
        instance = super().__new__(cls)
        if shape is not None:
            if isinstance(shape, Shape):
                shape_obj = shape
            else:
                shape_obj = Shape(shape)
            instance.set_shape(shape_obj)
        instance.set_style(apply_phong_lighting=False)
        return instance

    @lazy_basedata
    @staticmethod
    def _shape_() -> Shape:
        return Shape()

    @lazy_basedata
    @staticmethod
    def _stroke_mobjects_() -> list[StrokeMobject]:
        return []

    #@lazy_property
    #@staticmethod
    #def _geometry_o_() -> ShapeGeometry:
    #    return ShapeGeometry()

    @lazy_property
    @staticmethod
    def _geometry_(
        #geometry_o: ShapeGeometry,
        shape: Shape
    ) -> ShapeGeometry:
        return ShapeGeometry(shape)
        #geometry_o._shape_ = shape
        #return geometry_o

    #@lazy_basedata
    #@staticmethod
    #def _apply_phong_lighting_() -> bool:
    #    return False

    def set_shape(self, shape: Shape):
        self._shape_ = LazyData(shape)
        multi_line_string_data = LazyData(shape._multi_line_string_3d_)
        for stroke in self._stroke_mobjects_:
            stroke._multi_line_string_ = multi_line_string_data
        return self

    #def _set_fill_locally(
    #    self,
    #    *,
    #    color: ColorType | None = None,
    #    opacity: Real | None = None,
    #    apply_oit: bool | None = None,
    #    ambient_strength: Real | None = None,
    #    specular_strength: Real | None = None,
    #    shininess: Real | None = None,
    #    apply_phong_lighting: bool | None = None
    #):
    #    super()._set_style_locally(
    #        color=color,
    #        opacity=opacity,
    #        apply_oit=apply_oit,
    #        ambient_strength=ambient_strength,
    #        specular_strength=specular_strength,
    #        shininess=shininess,
    #        apply_phong_lighting=apply_phong_lighting
    #    )

    #def _add_stroke_locally(
    #    self,
    #    *,
    #    width: Real | None = None,
    #    single_sided: bool | None = None,
    #    has_linecap: bool | None = None,
    #    color: ColorType | None = None,
    #    opacity: Real | None = None,
    #    dilate: Real | None = None,
    #    apply_oit: bool | None = None
    #):
    #    stroke = StrokeMobject(self._shape_._multi_line_string_3d_)
    #    stroke.apply_transform(self._model_matrix_)
    #    stroke._set_style_locally(
    #        width=width,
    #        single_sided=single_sided,
    #        has_linecap=has_linecap,
    #        color=color,
    #        opacity=opacity,
    #        dilate=dilate,
    #        apply_oit=apply_oit
    #    )
    #    self._stroke_mobjects.append(stroke)
    #    self.add(stroke)
    #    return self

    #def _set_stroke_locally(
    #    self,
    #    *,
    #    index: int | None = None,
    #    width: Real | None = None,
    #    single_sided: bool | None = None,
    #    has_linecap: bool | None = None,
    #    color: ColorType | None = None,
    #    opacity: Real | None = None,
    #    dilate: Real | None = None,
    #    apply_oit: bool | None = None
    #):
    #    if self._stroke_mobjects:
    #        if index is None:
    #            index = 0
    #        self._stroke_mobjects[index]._set_style_locally(
    #            width=width,
    #            single_sided=single_sided,
    #            has_linecap=has_linecap,
    #            color=color,
    #            opacity=opacity,
    #            dilate=dilate,
    #            apply_oit=apply_oit
    #        )
    #    else:
    #        if index is not None:
    #            raise IndexError
    #        if any(param is not None for param in (
    #            width,
    #            single_sided,
    #            has_linecap,
    #            color,
    #            opacity,
    #            dilate,
    #            apply_oit
    #        )):
    #            self._add_stroke_locally(
    #                width=width,
    #                single_sided=single_sided,
    #                has_linecap=has_linecap,
    #                color=color,
    #                opacity=opacity,
    #                dilate=dilate,
    #                apply_oit=apply_oit
    #            )
    #    return self

    #def _set_style_locally(
    #    self,
    #    *,
    #    color: ColorType | None = None,
    #    opacity: Real | None = None,
    #    apply_oit: bool | None = None,
    #    ambient_strength: Real | None = None,
    #    specular_strength: Real | None = None,
    #    shininess: Real | None = None,
    #    apply_phong_lighting: bool | None = None,
    #    stroke_width: Real | None = None,
    #    stroke_single_sided: bool | None = None,
    #    stroke_has_linecap: bool | None = None,
    #    stroke_color: ColorType | None = None,
    #    stroke_opacity: Real | None = None,
    #    stroke_dilate: Real | None = None,
    #    stroke_apply_oit: bool | None = None
    #):
    #    self._set_fill_locally(
    #        color=color,
    #        opacity=opacity,
    #        apply_oit=apply_oit,
    #        ambient_strength=ambient_strength,
    #        specular_strength=specular_strength,
    #        shininess=shininess,
    #        apply_phong_lighting=apply_phong_lighting
    #    )
    #    self._set_stroke_locally(
    #        index=None,
    #        width=stroke_width,
    #        single_sided=stroke_single_sided,
    #        has_linecap=stroke_has_linecap,
    #        color=stroke_color,
    #        opacity=stroke_opacity,
    #        dilate=stroke_dilate,
    #        apply_oit=stroke_apply_oit
    #    )
    #    return self

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
        # TODO: almost completely redundant with MeshMobject.set_style
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        color_data = LazyData(color_component) if color_component is not None else None
        opacity_data = LazyData(opacity_component) if opacity_component is not None else None
        apply_oit_data = LazyData(apply_oit) if apply_oit is not None else \
            LazyData(True) if opacity_component is not None else None
        ambient_strength_data = LazyData(ambient_strength) if ambient_strength is not None else None
        specular_strength_data = LazyData(specular_strength) if specular_strength is not None else None
        shininess_data = LazyData(shininess) if shininess is not None else None
        apply_phong_lighting_data = LazyData(apply_phong_lighting) if apply_phong_lighting is not None else \
            LazyData(True) if any(param is not None for param in (
                ambient_strength,
                specular_strength,
                shininess
            )) else None
        for mobject in self.iter_descendants(broadcast=broadcast):
            if not isinstance(mobject, ShapeMobject):
                continue
            if color_data is not None:
                mobject._color_ = color_data
            if opacity_data is not None:
                mobject._opacity_ = opacity_data
            if apply_oit_data is not None:
                mobject._apply_oit_ = apply_oit_data
            if ambient_strength_data is not None:
                mobject._ambient_strength_ = ambient_strength_data
            if specular_strength_data is not None:
                mobject._specular_strength_ = specular_strength_data
            if shininess_data is not None:
                mobject._shininess_ = shininess_data
            if apply_phong_lighting_data is not None:
                mobject._apply_phong_lighting_ = apply_phong_lighting_data
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
        stroke_mobjects: list[StrokeMobject] = []
        for mobject in self.iter_descendants(broadcast=broadcast):
            if not isinstance(mobject, ShapeMobject):
                continue
            stroke = StrokeMobject(self._shape_._multi_line_string_3d_)
            stroke.apply_transform(self._model_matrix_)
            #mobject._add_stroke_locally(
            #    width=width,
            #    single_sided=single_sided,
            #    has_linecap=has_linecap,
            #    color=color,
            #    opacity=opacity,
            #    dilate=dilate,
            #    apply_oit=apply_oit
            #)
            stroke_mobjects.append(stroke)

        #self._stroke_mobjects.extend(stroke_mobjects)
        self._stroke_mobjects_ = LazyData(self._stroke_mobjects_ + stroke_mobjects)
        self.add(*stroke_mobjects)
        StrokeMobject().add(*stroke_mobjects).set_style(
            width=width,
            single_sided=single_sided,
            has_linecap=has_linecap,
            color=color,
            opacity=opacity,
            dilate=dilate,
            apply_oit=apply_oit,
            broadcast=broadcast
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
        if self._stroke_mobjects_:
            if index is None:
                index = 0
            stroke_mobjects = self._stroke_mobjects_[:]
            stroke_mobjects.insert(index, stroke_mobjects.pop(index).set_style(
                width=width,
                single_sided=single_sided,
                has_linecap=has_linecap,
                color=color,
                opacity=opacity,
                dilate=dilate,
                apply_oit=apply_oit,
                broadcast=broadcast
            ))
            self._stroke_mobjects_ = LazyData(stroke_mobjects)
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
                self.add_stroke(
                    width=width,
                    single_sided=single_sided,
                    has_linecap=has_linecap,
                    color=color,
                    opacity=opacity,
                    dilate=dilate,
                    apply_oit=apply_oit,
                    broadcast=broadcast
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
        self.set_fill(
            color=color,
            opacity=opacity,
            apply_oit=apply_oit,
            ambient_strength=ambient_strength,
            specular_strength=specular_strength,
            shininess=shininess,
            apply_phong_lighting=apply_phong_lighting,
            broadcast=broadcast
        )
        self.set_stroke(
            index=None,
            width=stroke_width,
            single_sided=stroke_single_sided,
            has_linecap=stroke_has_linecap,
            color=stroke_color,
            opacity=stroke_opacity,
            dilate=stroke_dilate,
            apply_oit=stroke_apply_oit,
            broadcast=broadcast
        )
        return self
