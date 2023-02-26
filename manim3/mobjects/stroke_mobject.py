__all__ = ["StrokeMobject"]


import itertools as it
from typing import Callable

import moderngl
import numpy as np

from ..cameras.camera import Camera
from ..custom_typing import (
    ColorType,
    Real,
    Vec3T,
    Vec3sT,
    VertexIndexType
)
from ..mobjects.mobject import Mobject
from ..rendering.glsl_buffers import (
    AttributesBuffer,
    IndexBuffer,
    UniformBlockBuffer
)
from ..rendering.vertex_array import (
    ContextState,
    VertexArray
)
from ..scenes.scene_config import SceneConfig
from ..utils.color import ColorUtils
from ..utils.lazy import (
    LazyWrapper,
    lazy_object,
    lazy_object_raw,
    lazy_property_raw,
    lazy_object_shared,
    lazy_property
)
from ..utils.shape import (
    LineString3D,
    LineStringKind,
    MultiLineString3D
)
from ..utils.space import SpaceUtils


class StrokeMobject(Mobject):
    __slots__ = ()

    def __init__(
        self,
        multi_line_string_3d: MultiLineString3D | None = None
    ):
        super().__init__()
        if multi_line_string_3d is not None:
            self._multi_line_string_3d_ = multi_line_string_3d

    @staticmethod
    def __winding_sign_key(
        winding_sign: bool
    ) -> bool:
        return winding_sign

    @lazy_object_shared(__winding_sign_key)
    @staticmethod
    def _winding_sign_() -> bool:
        return NotImplemented

    @lazy_object
    @staticmethod
    def _multi_line_string_3d_() -> MultiLineString3D:
        return MultiLineString3D()

    @lazy_object_raw
    @staticmethod
    def _width_() -> Real:
        # TODO: The unit mismatches by a factor of 5
        return 0.2

    @lazy_object_raw
    @staticmethod
    def _single_sided_() -> bool:
        return False

    @lazy_object_raw
    @staticmethod
    def _has_linecap_() -> bool:
        return True

    @lazy_object_raw
    @staticmethod
    def _color_() -> Vec3T:
        return np.ones(3)

    @lazy_object_raw
    @staticmethod
    def _opacity_() -> Real:
        return 1.0

    @lazy_object_raw
    @staticmethod
    def _dilate_() -> Real:
        return 0.0

    @lazy_property_raw
    @staticmethod
    def _local_sample_points_(
        _multi_line_string_3d_: MultiLineString3D
    ) -> Vec3sT:
        line_strings = _multi_line_string_3d_._children_
        if not line_strings:
            return np.zeros((0, 3))
        return np.concatenate([
            line_string._coords_.value
            for line_string in line_strings
        ])

    @lazy_property
    @staticmethod
    def _ub_stroke_(
        width: Real,
        color: Vec3T,
        opacity: Real,
        dilate: Real
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

    @lazy_property
    @staticmethod
    def _ub_winding_sign_(
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

    @lazy_property
    @staticmethod
    def _attributes_(
        _multi_line_string_3d_: MultiLineString3D
    ) -> AttributesBuffer:
        if not _multi_line_string_3d_._children_:
            position = np.zeros((0, 3))
        else:
            position = np.concatenate([
                line_string._coords_.value
                for line_string in _multi_line_string_3d_._children_
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

    @lazy_property_raw
    @staticmethod
    def _vertex_array_items_(
        _multi_line_string_3d_: MultiLineString3D,
        single_sided: bool,
        has_linecap: bool,
        _attributes_: AttributesBuffer
    ) -> list[tuple[VertexArray, list[str]]]:
        def get_vertex_array(index_getter: Callable[[int, LineStringKind], list[int]], mode: int) -> VertexArray:
            return VertexArray(
                attributes=_attributes_,
                index_buffer=IndexBuffer(
                    data=StrokeMobject._lump_index_from_getter(index_getter, _multi_line_string_3d_)
                ),
                mode=mode
            )

        subroutine_name = "single_sided" if single_sided else "both_sided"
        result: list[tuple[VertexArray, list[str]]] = [
            (get_vertex_array(StrokeMobject._line_index_getter, moderngl.LINES), [
                "#define STROKE_LINE",
                f"#define line_subroutine {subroutine_name}"
            ]),
            (get_vertex_array(StrokeMobject._join_index_getter, moderngl.TRIANGLES), [
                "#define STROKE_JOIN",
                f"#define join_subroutine {subroutine_name}"
            ])
        ]
        if has_linecap and not single_sided:
            result.extend([
                (get_vertex_array(StrokeMobject._cap_index_getter, moderngl.LINES), [
                    "#define STROKE_CAP"
                ]),
                (get_vertex_array(StrokeMobject._point_index_getter, moderngl.POINTS), [
                    "#define STROKE_POINT"
                ])
            ])
        return result

    @_vertex_array_items_.restocker
    @staticmethod
    def _vertex_array_items_restocker(
        vertex_array_items: list[tuple[VertexArray, list[str]]]
    ) -> None:
        for vertex_array, _ in vertex_array_items:
            vertex_array._restock()

    #@lazy_slot
    #@staticmethod
    #def _render_samples() -> int:
    #    return 4

    @classmethod
    def _lump_index_from_getter(
        cls,
        index_getter: Callable[[int, LineStringKind], list[int]],
        multi_line_string_3d: MultiLineString3D
    ) -> VertexIndexType:
        offset = 0
        index_arrays: list[VertexIndexType] = []
        for line_string in multi_line_string_3d._children_:
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
        #n_points = len(line_string._coords_.value)
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
        raise ValueError  # never

    @classmethod
    def _join_index_getter(
        cls,
        coords_len: int,
        kind: LineStringKind
    ) -> list[int]:
        if kind == LineStringKind.POINT:
            return []
        #n_points = len(line_string._coords_.value)
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
        raise ValueError  # never

    @classmethod
    def _cap_index_getter(
        cls,
        coords_len: int,
        kind: LineStringKind
    ) -> list[int]:
        if kind == LineStringKind.POINT:
            return []
        #n_points = len(line_string._coords_.value)
        if kind == LineStringKind.LINE_STRING:
            return [0, 1, coords_len - 1, coords_len - 2]
        if kind == LineStringKind.LINEAR_RING:
            return []
        raise ValueError  # never

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
        raise ValueError  # never

    def _render(
        self,
        scene_config: SceneConfig,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        # TODO: Is this already the best practice?
        self._winding_sign_ = self._calculate_winding_sign(scene_config._camera_)
        uniform_blocks = [
            scene_config._camera_._ub_camera_,
            self._ub_model_,
            self._ub_stroke_,
            self._ub_winding_sign_
        ]
        # Render color
        target_framebuffer.depth_mask = False
        for vertex_array, custom_macros in self._vertex_array_items_.value:
            vertex_array.render(
                shader_filename="stroke",
                custom_macros=custom_macros,
                texture_storages=[],
                uniform_blocks=uniform_blocks,
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
        for vertex_array, custom_macros in self._vertex_array_items_.value:
            vertex_array.render(
                shader_filename="stroke",
                custom_macros=custom_macros,
                texture_storages=[],
                uniform_blocks=uniform_blocks,
                framebuffer=target_framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.DEPTH_TEST
                )
            )
        target_framebuffer.color_mask = (True, True, True, True)

    def _calculate_winding_sign(
        self,
        camera: Camera
    ) -> bool:
        # TODO: The calculation here is somehow redundant with what shader does...
        area = 0.0
        transform = camera._projection_matrix_.value @ camera._view_matrix_.value @ self._model_matrix_.value
        for line_string in self._multi_line_string_3d_._children_:
            coords_2d = SpaceUtils.apply_affine(transform, line_string._coords_.value)[:, :2]
            area += np.cross(coords_2d, np.roll(coords_2d, -1, axis=0)).sum()
        return area * self._width_.value >= 0.0

    def set_style(
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
        width_value = LazyWrapper(width) if width is not None else None
        single_sided_value = LazyWrapper(single_sided) if single_sided is not None else None
        has_linecap_value = LazyWrapper(has_linecap) if has_linecap is not None else None
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        color_value = LazyWrapper(color_component) if color_component is not None else None
        opacity_value = LazyWrapper(opacity_component) if opacity_component is not None else None
        dilate_value = LazyWrapper(dilate) if dilate is not None else None
        apply_oit_value = LazyWrapper(apply_oit) if apply_oit is not None else \
            LazyWrapper(True) if any(param is not None for param in (
                opacity_component,
                dilate
            )) else None
        for mobject in self.iter_descendants(broadcast=broadcast):
            if not isinstance(mobject, StrokeMobject):
                continue
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
        return self
