__all__ = ["ShapeMobject"]


from typing import (
    Generator,
    Iterable
)

import shapely.geometry
import svgelements as se

from ..custom_typing import ColorType
from ..geometries.shape_geometry import ShapeGeometry
from ..lazy.core import LazyCollection
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.stroke_mobject import StrokeMobject
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

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _shape_(cls) -> Shape:
        return Shape()

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _geometry_(
        cls,
        _shape_: Shape
    ) -> ShapeGeometry:
        return ShapeGeometry(_shape_)

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _apply_phong_lighting_(cls) -> bool:
        return False

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _stroke_mobjects_(cls) -> LazyCollection[StrokeMobject]:
        return LazyCollection()

    def set_shape(
        self,
        shape: Shape
    ):
        self._shape_ = shape
        for stroke in self._stroke_mobjects_:
            stroke._multi_line_string_3d_ = shape._multi_line_string_3d_
        return self

    def adjust_stroke_shape(
        self,
        stroke_mobject: StrokeMobject
    ):
        stroke_mobject._model_matrix_ = self._model_matrix_
        stroke_mobject._multi_line_string_3d_ = self._shape_._multi_line_string_3d_
        return self

    def iter_shape_descendants(
        self,
        broadcast: bool = True
    ) -> "Generator[ShapeMobject, None, None]":
        for mobject in self.iter_descendants(broadcast=broadcast):
            if isinstance(mobject, ShapeMobject):
                yield mobject

    @classmethod
    def class_set_fill(
        cls,
        mobjects: "Iterable[ShapeMobject]",
        *,
        color: ColorType | None = None,
        opacity: float | None = None,
        apply_oit: bool | None = None,
        ambient_strength: float | None = None,
        specular_strength: float | None = None,
        shininess: float | None = None,
        apply_phong_lighting: bool | None = None
    ) -> None:
        MeshMobject.class_set_style(
            mobjects=mobjects,
            color=color,
            opacity=opacity,
            apply_oit=apply_oit,
            ambient_strength=ambient_strength,
            specular_strength=specular_strength,
            shininess=shininess,
            apply_phong_lighting=apply_phong_lighting
        )

    def set_fill(
        self,
        *,
        color: ColorType | None = None,
        opacity: float | None = None,
        apply_oit: bool | None = None,
        ambient_strength: float | None = None,
        specular_strength: float | None = None,
        shininess: float | None = None,
        apply_phong_lighting: bool | None = None,
        broadcast: bool = True
    ):
        self.class_set_fill(
            mobjects=self.iter_shape_descendants(broadcast=broadcast),
            color=color,
            opacity=opacity,
            apply_oit=apply_oit,
            ambient_strength=ambient_strength,
            specular_strength=specular_strength,
            shininess=shininess,
            apply_phong_lighting=apply_phong_lighting
        )
        return self

    @classmethod
    def class_add_stroke(
        cls,
        mobjects: "Iterable[ShapeMobject]",
        *,
        width: float | None = None,
        single_sided: bool | None = None,
        has_linecap: bool | None = None,
        color: ColorType | None = None,
        opacity: float | None = None,
        dilate: float | None = None,
        apply_oit: bool | None = None
    ) -> None:
        if all(param is None for param in (
            width,
            single_sided,
            has_linecap,
            color,
            opacity,
            dilate,
            apply_oit
        )):
            return

        stroke_mobjects: list[StrokeMobject] = []
        for mobject in mobjects:
            stroke = StrokeMobject()
            mobject.adjust_stroke_shape(stroke)
            stroke_mobjects.append(stroke)
            mobject._stroke_mobjects_.add(stroke)
            mobject.add(stroke)

        StrokeMobject.class_set_style(
            mobjects=stroke_mobjects,
            width=width,
            single_sided=single_sided,
            has_linecap=has_linecap,
            color=color,
            opacity=opacity,
            dilate=dilate,
            apply_oit=apply_oit
        )

    def add_stroke(
        self,
        *,
        width: float | None = None,
        single_sided: bool | None = None,
        has_linecap: bool | None = None,
        color: ColorType | None = None,
        opacity: float | None = None,
        dilate: float | None = None,
        apply_oit: bool | None = None,
        broadcast: bool = True
    ):
        self.class_add_stroke(
            mobjects=self.iter_shape_descendants(broadcast=broadcast),
            width=width,
            single_sided=single_sided,
            has_linecap=has_linecap,
            color=color,
            opacity=opacity,
            dilate=dilate,
            apply_oit=apply_oit
        )
        return self

    @classmethod
    def class_set_stroke(
        cls,
        mobjects: "Iterable[ShapeMobject]",
        *,
        index: int | None = None,
        width: float | None = None,
        single_sided: bool | None = None,
        has_linecap: bool | None = None,
        color: ColorType | None = None,
        opacity: float | None = None,
        dilate: float | None = None,
        apply_oit: bool | None = None
    ) -> None:
        index_is_none = index is None
        index = index if index is not None else 0
        stroke_mobjects: list[StrokeMobject] = []
        shape_mobjects: list[ShapeMobject] = []
        for mobject in mobjects:
            if mobject._stroke_mobjects_:
                stroke_mobjects.append(mobject._stroke_mobjects_[index])
            else:
                if not index_is_none:
                    raise IndexError
                shape_mobjects.append(mobject)
        StrokeMobject.class_set_style(
            mobjects=stroke_mobjects,
            width=width,
            single_sided=single_sided,
            has_linecap=has_linecap,
            color=color,
            opacity=opacity,
            dilate=dilate,
            apply_oit=apply_oit
        )
        cls.class_add_stroke(
            mobjects=shape_mobjects,
            width=width,
            single_sided=single_sided,
            has_linecap=has_linecap,
            color=color,
            opacity=opacity,
            dilate=dilate,
            apply_oit=apply_oit
        )

    def set_stroke(
        self,
        *,
        index: int | None = None,
        width: float | None = None,
        single_sided: bool | None = None,
        has_linecap: bool | None = None,
        color: ColorType | None = None,
        opacity: float | None = None,
        dilate: float | None = None,
        apply_oit: bool | None = None,
        broadcast: bool = True
    ):
        self.class_set_stroke(
            mobjects=self.iter_shape_descendants(broadcast=broadcast),
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

    @classmethod
    def class_set_style(
        cls,
        mobjects: "Iterable[ShapeMobject]",
        *,
        color: ColorType | None = None,
        opacity: float | None = None,
        apply_oit: bool | None = None,
        ambient_strength: float | None = None,
        specular_strength: float | None = None,
        shininess: float | None = None,
        apply_phong_lighting: bool | None = None,
        stroke_index: int | None = None,
        stroke_width: float | None = None,
        stroke_single_sided: bool | None = None,
        stroke_has_linecap: bool | None = None,
        stroke_color: ColorType | None = None,
        stroke_opacity: float | None = None,
        stroke_dilate: float | None = None,
        stroke_apply_oit: bool | None = None
    ) -> None:
        cls.class_set_fill(
            mobjects=mobjects,
            color=color,
            opacity=opacity,
            apply_oit=apply_oit,
            ambient_strength=ambient_strength,
            specular_strength=specular_strength,
            shininess=shininess,
            apply_phong_lighting=apply_phong_lighting
        )
        cls.class_set_stroke(
            mobjects=mobjects,
            index=stroke_index,
            width=stroke_width,
            single_sided=stroke_single_sided,
            has_linecap=stroke_has_linecap,
            color=stroke_color,
            opacity=stroke_opacity,
            dilate=stroke_dilate,
            apply_oit=stroke_apply_oit
        )

    def set_style(
        self,
        *,
        color: ColorType | None = None,
        opacity: float | None = None,
        apply_oit: bool | None = None,
        ambient_strength: float | None = None,
        specular_strength: float | None = None,
        shininess: float | None = None,
        apply_phong_lighting: bool | None = None,
        stroke_index: int | None = None,
        stroke_width: float | None = None,
        stroke_single_sided: bool | None = None,
        stroke_has_linecap: bool | None = None,
        stroke_color: ColorType | None = None,
        stroke_opacity: float | None = None,
        stroke_dilate: float | None = None,
        stroke_apply_oit: bool | None = None,
        broadcast: bool = True
    ):
        self.class_set_style(
            mobjects=self.iter_shape_descendants(broadcast=broadcast),
            color=color,
            opacity=opacity,
            apply_oit=apply_oit,
            ambient_strength=ambient_strength,
            specular_strength=specular_strength,
            shininess=shininess,
            apply_phong_lighting=apply_phong_lighting,
            stroke_index=stroke_index,
            stroke_width=stroke_width,
            stroke_single_sided=stroke_single_sided,
            stroke_has_linecap=stroke_has_linecap,
            stroke_color=stroke_color,
            stroke_opacity=stroke_opacity,
            stroke_dilate=stroke_dilate,
            stroke_apply_oit=stroke_apply_oit
        )
        return self
