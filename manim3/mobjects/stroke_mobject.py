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
from ..rendering.glsl_variables import (
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
    NewData,
    lazy_basedata,
    lazy_basedata_cached,
    lazy_property,
    lazy_slot
)
from ..utils.shape import (
    LineString3D,
    MultiLineString3D
)
from ..utils.space import SpaceUtils


class StrokeMobject(Mobject):
    __slots__ = ()

    def __init__(self, multi_line_string_3d: MultiLineString3D | None = None):
        super().__init__()
        if multi_line_string_3d is not None:
            self._multi_line_string_3d_ = NewData(multi_line_string_3d)

    @staticmethod
    def __winding_sign_cacher(
        winding_sign: bool
    ) -> bool:
        return winding_sign

    @lazy_basedata_cached(__winding_sign_cacher)
    @staticmethod
    def _winding_sign_() -> bool:
        return NotImplemented

    @lazy_basedata
    @staticmethod
    def _multi_line_string_3d_() -> MultiLineString3D:
        return MultiLineString3D()

    @lazy_basedata
    @staticmethod
    def _width_() -> Real:
        # TODO: The unit mismatches by a factor of 5
        return 0.2

    @lazy_basedata
    @staticmethod
    def _single_sided_() -> bool:
        return False

    @lazy_basedata
    @staticmethod
    def _has_linecap_() -> bool:
        return True

    @lazy_basedata
    @staticmethod
    def _color_() -> Vec3T:
        return np.ones(3)

    @lazy_basedata
    @staticmethod
    def _opacity_() -> Real:
        return 1.0

    @lazy_basedata
    @staticmethod
    def _dilate_() -> Real:
        return 0.0

    @lazy_property
    @staticmethod
    def _local_sample_points_(multi_line_string_3d: MultiLineString3D) -> Vec3sT:
        line_strings = multi_line_string_3d._children_
        if not line_strings:
            return np.zeros((0, 3))
        return np.concatenate([
            line_string._coords_
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
        multi_line_string_3d: MultiLineString3D
    ) -> AttributesBuffer:
        if not multi_line_string_3d._children_:
            position = np.zeros((0, 3))
        else:
            position = np.concatenate([
                line_string._coords_
                for line_string in multi_line_string_3d._children_
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

    @lazy_property
    @staticmethod
    def _vertex_array_items_(
        multi_line_string_3d: MultiLineString3D,
        single_sided: bool,
        has_linecap: bool,
        attributes: AttributesBuffer
    ) -> list[tuple[VertexArray, list[str]]]:
        def get_vertex_array(index_getter: Callable[[LineString3D], list[int]], mode: int) -> VertexArray:
            return VertexArray(
                attributes=attributes,
                index_buffer=IndexBuffer(
                    data=StrokeMobject._lump_index_from_getter(index_getter, multi_line_string_3d)
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

    @lazy_slot
    @staticmethod
    def _render_samples() -> int:
        return 4

    @classmethod
    def _lump_index_from_getter(
        cls,
        index_getter: Callable[[LineString3D], list[int]],
        multi_line_string_3d: MultiLineString3D
    ) -> VertexIndexType:
        offset = 0
        index_arrays: list[VertexIndexType] = []
        for line_string in multi_line_string_3d._children_:
            index_arrays.append(np.array(index_getter(line_string), dtype=np.uint32) + offset)
            offset += len(line_string._coords_)
        if not index_arrays:
            return np.zeros(0, dtype=np.uint32)
        return np.concatenate(index_arrays, dtype=np.uint32)

    @classmethod
    def _line_index_getter(cls, line_string: LineString3D) -> list[int]:
        if line_string._kind_ == "point":
            return []
        n_points = len(line_string._coords_)
        if line_string._kind_ == "line_string":
            # (0, 1, 1, 2, ..., n-2, n-1)
            return list(it.chain(*zip(*(
                range(i, n_points - 1 + i)
                for i in range(2)
            ))))
        if line_string._kind_ == "linear_ring":
            return list(it.chain(*zip(*(
                np.roll(range(n_points - 1), -i)
                for i in range(2)
            ))))
        raise ValueError  # never

    @classmethod
    def _join_index_getter(cls, line_string: LineString3D) -> list[int]:
        if line_string._kind_ == "point":
            return []
        n_points = len(line_string._coords_)
        if line_string._kind_ == "line_string":
            # (0, 1, 2, 1, 2, 3, ..., n-3, n-2, n-1)
            return list(it.chain(*zip(*(
                range(i, n_points - 2 + i)
                for i in range(3)
            ))))
        if line_string._kind_ == "linear_ring":
            return list(it.chain(*zip(*(
                np.roll(range(n_points - 1), -i)
                for i in range(3)
            ))))
        raise ValueError  # never

    @classmethod
    def _cap_index_getter(cls, line_string: LineString3D) -> list[int]:
        if line_string._kind_ in "point":
            return []
        n_points = len(line_string._coords_)
        if line_string._kind_ == "line_string":
            return [0, 1, n_points - 1, n_points - 2]
        if line_string._kind_ == "linear_ring":
            return []
        raise ValueError  # never

    @classmethod
    def _point_index_getter(cls, line_string: LineString3D) -> list[int]:
        if line_string._kind_ in "point":
            return [0]
        if line_string._kind_ == "line_string":
            return []
        if line_string._kind_ == "linear_ring":
            return []
        raise ValueError  # never

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        # TODO: Is this already the best practice?
        self._winding_sign_ = self._calculate_winding_sign(scene_config._camera)
        uniform_blocks = [
            scene_config._camera._ub_camera_,
            self._ub_model_,
            self._ub_stroke_,
            self._ub_winding_sign_
        ]
        # Render color
        target_framebuffer.depth_mask = False
        for vertex_array, custom_macros in self._vertex_array_items_:
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
        for vertex_array, custom_macros in self._vertex_array_items_:
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

    def _calculate_winding_sign(self, camera: Camera) -> bool:
        # TODO: The calculation here is somehow redundant with what shader does...
        area = 0.0
        transform = camera._projection_matrix_ @ camera._view_matrix_ @ self._model_matrix_
        for line_string in self._multi_line_string_3d_._children_:
            coords_2d = SpaceUtils.apply_affine(transform, line_string._coords_)[:, :2]
            area += np.cross(coords_2d, np.roll(coords_2d, -1, axis=0)).sum()
        return area * self._width_ >= 0.0

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
        width_data = NewData(width) if width is not None else None
        single_sided_data = NewData(single_sided) if single_sided is not None else None
        has_linecap_data = NewData(has_linecap) if has_linecap is not None else None
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        color_data = NewData(color_component) if color_component is not None else None
        opacity_data = NewData(opacity_component) if opacity_component is not None else None
        dilate_data = NewData(dilate) if dilate is not None else None
        apply_oit = apply_oit if apply_oit is not None else \
            True if any(param is not None for param in (
                opacity_component,
                dilate
            )) else None
        for mobject in self.iter_descendants(broadcast=broadcast):
            if not isinstance(mobject, StrokeMobject):
                continue
            if width_data is not None:
                mobject._width_ = width_data
            if single_sided_data is not None:
                mobject._single_sided_ = single_sided_data
            if has_linecap_data is not None:
                mobject._has_linecap_ = has_linecap_data
            if color_data is not None:
                mobject._color_ = color_data
            if opacity_data is not None:
                mobject._opacity_ = opacity_data
            if dilate_data is not None:
                mobject._dilate_ = dilate_data
            if apply_oit is not None:
                mobject._apply_oit = apply_oit
        return self
