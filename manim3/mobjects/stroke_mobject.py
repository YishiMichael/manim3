__all__ = ["StrokeMobject"]


import itertools as it
from typing import (
    Callable,
    Generator,
    Iterable
)

import moderngl
import numpy as np

from ..custom_typing import (
    ColorType,
    Mat4T,
    Vec2sT,
    Vec3T,
    Vec3sT,
    VertexIndexType
)
from ..lazy.core import (
    LazyDynamicVariableDescriptor,
    LazyWrapper
)
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..mobjects.mobject import Mobject
from ..rendering.context import ContextState
from ..rendering.gl_buffer import (
    AttributesBuffer,
    IndexBuffer,
    UniformBlockBuffer
)
from ..rendering.vertex_array import (
    IndexedAttributesBuffer,
    VertexArray
)
from ..utils.color import ColorUtils
from ..utils.shape import (
    LineStringKind,
    MultiLineString
)
from ..utils.space import SpaceUtils


class StrokeMobject(Mobject):
    __slots__ = ()

    def __init__(
        self,
        multi_line_string: MultiLineString | None = None
    ) -> None:
        super().__init__()
        if multi_line_string is not None:
            self._multi_line_string_ = multi_line_string

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _multi_line_string_(cls) -> MultiLineString:
        return MultiLineString()

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _width_(cls) -> float:
        # TODO: The unit mismatches by a factor of 5.
        return 0.2

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _single_sided_(cls) -> bool:
        return False

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _has_linecap_(cls) -> bool:
        return True

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _dilate_(cls) -> float:
        return 0.0

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _winding_sign_(
        cls,
        scene_config__camera__projection_matrix: Mat4T,
        scene_config__camera__view_matrix: Mat4T,
        model_matrix: Mat4T,
        multi_line_string__line_strings__coords: list[Vec3sT],
        width: float
    ) -> bool:
        # TODO: The calculation here is somehow redundant with what shader does...

        def get_signed_area(
            coords: Vec2sT
        ) -> float:
            return np.cross(coords, np.roll(coords, -1, axis=0)).sum() / 2.0

        transform = scene_config__camera__projection_matrix @ scene_config__camera__view_matrix @ model_matrix
        area = sum(
            get_signed_area(SpaceUtils.decrease_dimension(SpaceUtils.apply_affine(transform, coords)))
            for coords in multi_line_string__line_strings__coords
        )
        return bool(area * width >= 0.0)

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _local_sample_points_(
        cls,
        _multi_line_string_: MultiLineString
    ) -> Vec3sT:
        line_strings = _multi_line_string_._line_strings_
        if not line_strings:
            return np.zeros((0, 3))
        return np.concatenate([
            line_string._coords_.value
            for line_string in line_strings
        ])

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _ub_stroke_(
        cls,
        width: float,
        color: Vec3T,
        opacity: float,
        dilate: float
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_stroke",
            fields=[
                "float u_width",
                "vec4 u_color",
                "float u_dilate"
            ],
            data={
                "u_width": np.array(abs(width)),
                "u_color": np.append(color, opacity),
                "u_dilate": np.array(dilate)
            }
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _ub_winding_sign_(
        cls,
        winding_sign: bool
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_winding_sign",
            fields=[
                "float u_winding_sign"
            ],
            data={
                "u_winding_sign": np.array(1.0 if winding_sign else -1.0)
            }
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _attributes_buffer_(
        cls,
        _multi_line_string_: MultiLineString
    ) -> AttributesBuffer:
        if not _multi_line_string_._line_strings_:
            position = np.zeros((0, 3))
        else:
            position = np.concatenate([
                line_string._coords_.value
                for line_string in _multi_line_string_._line_strings_
            ])
        return AttributesBuffer(
            fields=[
                "vec3 in_position"
            ],
            num_vertex=len(position),
            data={
                "in_position": position
            }
        )

    @Lazy.property(LazyMode.COLLECTION)
    @classmethod
    def _vertex_arrays_(
        cls,
        _scene_config__camera__ub_camera_: UniformBlockBuffer,
        _ub_model_: UniformBlockBuffer,
        _ub_stroke_: UniformBlockBuffer,
        _ub_winding_sign_: UniformBlockBuffer,
        _attributes_buffer_: AttributesBuffer,
        _multi_line_string_: MultiLineString,
        single_sided: bool,
        has_linecap: bool
    ) -> list[VertexArray]:
        uniform_blocks = [
            _scene_config__camera__ub_camera_,
            _ub_model_,
            _ub_stroke_,
            _ub_winding_sign_
        ]

        def get_vertex_array(
            index_getter: Callable[[int, LineStringKind], list[int]],
            mode: int,
            custom_macros: list[str]
        ) -> VertexArray:
            return VertexArray(
                shader_filename="stroke",
                custom_macros=custom_macros,
                uniform_blocks=uniform_blocks,
                indexed_attributes_buffer=IndexedAttributesBuffer(
                    attributes_buffer=_attributes_buffer_,
                    index_buffer=IndexBuffer(
                        data=cls._lump_index_from_getter(index_getter, _multi_line_string_)
                    ),
                    mode=mode
                )
            )

        subroutine_name = "single_sided" if single_sided else "both_sided"
        vertex_arrays = [
            get_vertex_array(cls._line_index_getter, moderngl.LINES, [
                "#define STROKE_LINE",
                f"#define line_subroutine {subroutine_name}"
            ]),
            get_vertex_array(cls._join_index_getter, moderngl.TRIANGLES, [
                "#define STROKE_JOIN",
                f"#define join_subroutine {subroutine_name}"
            ])
        ]
        if has_linecap and not single_sided:
            vertex_arrays.extend([
                get_vertex_array(cls._cap_index_getter, moderngl.LINES, [
                    "#define STROKE_CAP"
                ]),
                get_vertex_array(cls._point_index_getter, moderngl.POINTS, [
                    "#define STROKE_POINT"
                ])
            ])
        return vertex_arrays

    @classmethod
    def _lump_index_from_getter(
        cls,
        index_getter: Callable[[int, LineStringKind], list[int]],
        multi_line_string: MultiLineString
    ) -> VertexIndexType:
        offset = 0
        index_arrays: list[VertexIndexType] = []
        for line_string in multi_line_string._line_strings_:
            coords_len = len(line_string._coords_.value)
            kind = line_string._kind_.value
            index_arrays.append(np.array(index_getter(coords_len, kind), dtype=np.uint32) + offset)
            offset += coords_len
        if not index_arrays:
            return np.zeros(0, dtype=np.uint32)
        return np.concatenate(index_arrays, dtype=np.uint32)

    @classmethod
    def _line_index_getter(
        cls,
        coords_len: int,
        kind: LineStringKind
    ) -> list[int]:
        if kind == LineStringKind.POINT:
            return []
        if kind == LineStringKind.LINE_STRING:
            # (0, 1, 1, 2, ..., n-2, n-1)
            return list(it.chain(*zip(*(
                range(i, coords_len - 1 + i)
                for i in range(2)
            ))))
        if kind == LineStringKind.LINEAR_RING:
            return list(it.chain(*zip(*(
                np.roll(range(coords_len - 1), -i)
                for i in range(2)
            ))))

    @classmethod
    def _join_index_getter(
        cls,
        coords_len: int,
        kind: LineStringKind
    ) -> list[int]:
        if kind == LineStringKind.POINT:
            return []
        if kind == LineStringKind.LINE_STRING:
            # (0, 1, 2, 1, 2, 3, ..., n-3, n-2, n-1)
            return list(it.chain(*zip(*(
                range(i, coords_len - 2 + i)
                for i in range(3)
            ))))
        if kind == LineStringKind.LINEAR_RING:
            return list(it.chain(*zip(*(
                np.roll(range(coords_len - 1), -i)
                for i in range(3)
            ))))

    @classmethod
    def _cap_index_getter(
        cls,
        coords_len: int,
        kind: LineStringKind
    ) -> list[int]:
        if kind == LineStringKind.POINT:
            return []
        if kind == LineStringKind.LINE_STRING:
            return [0, 1, coords_len - 1, coords_len - 2]
        if kind == LineStringKind.LINEAR_RING:
            return []

    @classmethod
    def _point_index_getter(
        cls,
        coords_len: int,
        kind: LineStringKind
    ) -> list[int]:
        if kind == LineStringKind.POINT:
            return [0]
        if kind == LineStringKind.LINE_STRING:
            return []
        if kind == LineStringKind.LINEAR_RING:
            return []

    def _render(
        self,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        # TODO: Is this already the best practice?
        # Render color
        target_framebuffer.depth_mask = False
        for vertex_array in self._vertex_arrays_:
            vertex_array.render(
                framebuffer=target_framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.BLEND,
                    blend_func=moderngl.ADDITIVE_BLENDING,
                    blend_equation=moderngl.MAX
                )
            )
        target_framebuffer.depth_mask = True
        # Render depth
        target_framebuffer.color_mask = (False, False, False, False)
        for vertex_array in self._vertex_arrays_:
            vertex_array.render(
                framebuffer=target_framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.DEPTH_TEST
                )
            )
        target_framebuffer.color_mask = (True, True, True, True)

    def iter_stroke_descendants(
        self,
        broadcast: bool = True
    ) -> "Generator[StrokeMobject, None, None]":
        for mobject in self.iter_descendants(broadcast=broadcast):
            if isinstance(mobject, StrokeMobject):
                yield mobject

    @classmethod
    def class_set_style(
        cls,
        mobjects: "Iterable[StrokeMobject]",
        *,
        width: float | None = None,
        single_sided: bool | None = None,
        has_linecap: bool | None = None,
        color: ColorType | None = None,
        opacity: float | None = None,
        dilate: float | None = None,
        apply_oit: bool | None = None
    ) -> None:
        width_value = LazyWrapper(width) if width is not None else None
        single_sided_value = LazyWrapper(single_sided) if single_sided is not None else None
        has_linecap_value = LazyWrapper(has_linecap) if has_linecap is not None else None
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        color_value = LazyWrapper(color_component) if color_component is not None else None
        opacity_value = LazyWrapper(opacity_component) if opacity_component is not None else None
        dilate_value = LazyWrapper(dilate) if dilate is not None else None
        apply_oit_value = apply_oit if apply_oit is not None else \
            True if any(param is not None for param in (
                opacity_component,
                dilate
            )) else None
        for mobject in mobjects:
            if width_value is not None:
                mobject._width_ = width_value
            if single_sided_value is not None:
                mobject._single_sided_ = single_sided_value
            if has_linecap_value is not None:
                mobject._has_linecap_ = has_linecap_value
            if color_value is not None:
                mobject._color_ = color_value
            if opacity_value is not None:
                mobject._opacity_ = opacity_value
            if dilate_value is not None:
                mobject._dilate_ = dilate_value
            if apply_oit_value is not None:
                mobject._apply_oit_ = apply_oit_value

    def set_style(
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
        self.class_set_style(
            mobjects=self.iter_stroke_descendants(broadcast=broadcast),
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
    def class_concatenate(
        cls,
        *mobjects: "StrokeMobject"
    ) -> "StrokeMobject":
        if not mobjects:
            return StrokeMobject()
        result = mobjects[0]._copy()
        for descriptor in cls._LAZY_VARIABLE_DESCRIPTORS:
            if isinstance(descriptor, LazyDynamicVariableDescriptor):
                continue
            if descriptor is cls._multi_line_string_:
                continue
            assert all(
                descriptor.__get__(result) is descriptor.__get__(mobject)
                for mobject in mobjects
            )
        result._multi_line_string_ = MultiLineString.concatenate(
            mobject._multi_line_string_
            for mobject in mobjects
        )
        return result
