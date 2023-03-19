__all__ = ["ShapeMobject"]


from typing import (
    Generator,
    Iterable
)

from ..custom_typing import ColorType
from ..geometries.shape_geometry import ShapeGeometry
from ..lazy.core import LazyDynamicVariableDescriptor
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
        shape: Shape | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            self.set_shape(shape)
        self.set_style(apply_phong_lighting=False)

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _shape_(cls) -> Shape:
        return Shape()

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _apply_phong_lighting_(cls) -> bool:
        return False

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _stroke_mobjects_(cls) -> list[StrokeMobject]:
        return []

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _geometry_(
        cls,
        _shape_: Shape
    ) -> ShapeGeometry:
        return ShapeGeometry(_shape_)

    def set_shape(
        self,
        shape: Shape
    ):
        self._shape_ = shape
        for stroke in self._stroke_mobjects_:
            stroke._multi_line_string_ = shape._multi_line_string_
        return self

    def adjust_stroke_shape(
        self,
        stroke_mobject: StrokeMobject
    ):
        stroke_mobject._model_matrix_ = self._model_matrix_
        stroke_mobject._multi_line_string_ = self._shape_._multi_line_string_
        return self

    def add_stroke_mobject(
        self,
        stroke_mobject: StrokeMobject
    ):
        self._stroke_mobjects_.add(stroke_mobject)
        self.add(stroke_mobject)
        return self

    def iter_shape_children(self) -> "Generator[ShapeMobject, None, None]":
        for mobject in self.iter_children():
            if isinstance(mobject, ShapeMobject):
                yield mobject

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
            mobject.add_stroke_mobject(stroke)
            stroke_mobjects.append(stroke)

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

    @classmethod
    def class_concatenate(
        cls,
        *mobjects: "ShapeMobject"
    ) -> "ShapeMobject":
        if not mobjects:
            return cls()
        result = mobjects[0]._copy()
        for descriptor in cls._LAZY_VARIABLE_DESCRIPTORS:
            if isinstance(descriptor, LazyDynamicVariableDescriptor):
                continue
            if descriptor is cls._shape_:
                continue
            assert all(
                descriptor.__get__(result) is descriptor.__get__(mobject)
                for mobject in mobjects
            )
        result._shape_ = Shape.concatenate(
            mobject._shape_
            for mobject in mobjects
        )

        stroke_mobjects = [
            StrokeMobject.class_concatenate(*zipped_stroke_mobjects)
            for zipped_stroke_mobjects in zip(*(
                mobject._stroke_mobjects_
                for mobject in mobjects
            ), strict=True)
        ]
        result._stroke_mobjects_ = stroke_mobjects
        result.clear()
        result.add(*stroke_mobjects)
        return result

    def concatenate(self) -> "ShapeMobject":
        return self.class_concatenate(*self.iter_shape_children())
