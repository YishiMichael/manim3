__all__ = ["ShapeMobject"]


import moderngl
import shapely.geometry
import svgelements as se

from ..custom_typing import (
    ColorType,
    Real
)
from ..geometries.shape_geometry import ShapeGeometry
from ..lazy.core import LazyCollection
from ..lazy.interfaces import (
    lazy_collection,
    lazy_object,
    lazy_property
)
from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.stroke_mobject import StrokeMobject
from ..utils.color import ColorUtils
from ..utils.scene_config import SceneConfig
from ..utils.shape import Shape


class ShapeMobject(MeshMobject):
    __slots__ = ()

    def __init__(
        self,
        shape: Shape | shapely.geometry.base.BaseGeometry | se.Shape | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            if isinstance(shape, Shape):
                shape_obj = shape
            else:
                shape_obj = Shape(shape)
            self.set_shape(shape_obj)
        self.set_style(apply_phong_lighting=False)

    @lazy_object
    @classmethod
    def _shape_(cls) -> Shape:
        return Shape()

    @lazy_property
    @classmethod
    def _geometry_(
        cls,
        _shape_: Shape
    ) -> ShapeGeometry:
        return ShapeGeometry(_shape_)

    #@lazy_value
    #@classmethod
    #def _apply_phong_lighting() -> bool:
    #    return False

    @lazy_collection
    @classmethod
    def _stroke_mobjects_(cls) -> LazyCollection[StrokeMobject]:
        return LazyCollection()

    def _render(
        self,
        scene_config: SceneConfig,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        super()._render(scene_config, target_framebuffer)
        for stroke_mobject in self._stroke_mobjects_:
            stroke_mobject._model_matrix_ = self._model_matrix_  # TODO: should it live here???
            stroke_mobject._render(scene_config, target_framebuffer)

    def set_shape(
        self,
        shape: Shape
    ):
        self._shape_ = shape
        for stroke in self._stroke_mobjects_:
            stroke._multi_line_string_3d_ = shape._multi_line_string_3d_
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
        # TODO: almost completely redundant with MeshMobject.set_style
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        color_value = color_component if color_component is not None else None
        opacity_value = opacity_component if opacity_component is not None else None
        apply_oit_value = apply_oit if apply_oit is not None else \
            True if opacity_component is not None else None
        ambient_strength_value = ambient_strength if ambient_strength is not None else None
        specular_strength_value = specular_strength if specular_strength is not None else None
        shininess_value = shininess if shininess is not None else None
        apply_phong_lighting_value = apply_phong_lighting if apply_phong_lighting is not None else \
            True if any(param is not None for param in (
                ambient_strength,
                specular_strength,
                shininess
            )) else None
        for mobject in self.iter_descendants(broadcast=broadcast):
            if not isinstance(mobject, ShapeMobject):
                continue
            if color_value is not None:
                mobject._color_ = color_value
            if opacity_value is not None:
                mobject._opacity_ = opacity_value
            if apply_oit_value is not None:
                mobject._apply_oit_ = apply_oit_value
            if ambient_strength_value is not None:
                mobject._ambient_strength_ = ambient_strength_value
            if specular_strength_value is not None:
                mobject._specular_strength_ = specular_strength_value
            if shininess_value is not None:
                mobject._shininess_ = shininess_value
            if apply_phong_lighting_value is not None:
                mobject._apply_phong_lighting_ = apply_phong_lighting_value
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
            #stroke.apply_transform(self._model_matrix_)
            stroke_mobjects.append(stroke)
            mobject._stroke_mobjects_.add(stroke)

        StrokeMobject().add(*stroke_mobjects).set_style(
            width=width,
            single_sided=single_sided,
            has_linecap=has_linecap,
            color=color,
            opacity=opacity,
            dilate=dilate,
            apply_oit=apply_oit,
            broadcast=True
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
            if mobject._stroke_mobjects_:
                if index is None:
                    index = 0
                mobject._stroke_mobjects_[index].set_style(
                    width=width,
                    single_sided=single_sided,
                    has_linecap=has_linecap,
                    color=color,
                    opacity=opacity,
                    dilate=dilate,
                    apply_oit=apply_oit,
                    broadcast=False
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
                    mobject.add_stroke(
                        width=width,
                        single_sided=single_sided,
                        has_linecap=has_linecap,
                        color=color,
                        opacity=opacity,
                        dilate=dilate,
                        apply_oit=apply_oit,
                        broadcast=False
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
