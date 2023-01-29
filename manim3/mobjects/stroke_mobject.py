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
from ..utils.color import ColorUtils
from ..utils.lazy import (
    lazy_property,
    lazy_property_writable
)
from ..utils.render_procedure import (
    AttributesBuffer,
    IndexBuffer,
    RenderProcedure,
    UniformBlockBuffer
)
from ..utils.scene_config import SceneConfig
from ..utils.shape import (
    LineString3D,
    MultiLineString3D
)


class StrokeMobject(Mobject):
    def __init__(self, multi_line_string: MultiLineString3D):
        super().__init__()
        self._multi_line_string_ = multi_line_string

    @lazy_property_writable
    @staticmethod
    def _multi_line_string_() -> MultiLineString3D:
        return NotImplemented

    @lazy_property_writable
    @staticmethod
    def _width_() -> Real:
        return 0.04

    @lazy_property_writable
    @staticmethod
    def _single_sided_() -> bool:
        return False

    @lazy_property_writable
    @staticmethod
    def _has_linecap_() -> bool:
        return True

    @lazy_property_writable
    @staticmethod
    def _color_() -> Vec3T:
        return np.ones(3)

    @lazy_property_writable
    @staticmethod
    def _opacity_() -> Real:
        return 1.0

    @lazy_property_writable
    @staticmethod
    def _dilate_() -> Real:
        return 0.0

    @lazy_property
    @staticmethod
    def _ub_stroke_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer("ub_stroke", [
            "float u_width",
            "vec4 u_color",
            "float u_dilate"
        ])

    @lazy_property
    @staticmethod
    def _ub_stroke_(
        ub_stroke_o: UniformBlockBuffer,
        width: Real,
        color: Vec3T,
        opacity: Real,
        dilate: Real
    ) -> UniformBlockBuffer:
        return ub_stroke_o.write({
            "u_width": np.array(width),
            "u_color": np.append(color, opacity),
            "u_dilate": np.array(dilate)
        })

    @lazy_property
    @staticmethod
    def _ub_winding_sign_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer("ub_winding_sign", [
            "float u_winding_sign"
        ])

    @lazy_property
    @staticmethod
    def _attributes_o_() -> AttributesBuffer:
        return AttributesBuffer([
            "vec3 in_position"
        ])

    @lazy_property
    @staticmethod
    def _attributes_(
        attributes_o: AttributesBuffer,
        multi_line_string: MultiLineString3D
    ) -> AttributesBuffer:
        if not multi_line_string._children_:
            position = np.zeros((0, 3))
        else:
            position = np.concatenate([
                line_string._coords_
                for line_string in multi_line_string._children_
            ])
        return attributes_o.write({
            "in_position": position
        })

    @classmethod
    def _lump_index_from_getter(
        cls,
        index_getter: Callable[[LineString3D], list[int]],
        multi_line_string: MultiLineString3D
    ) -> VertexIndexType:
        offset = 0
        index_arrays: list[VertexIndexType] = []
        for line_string in multi_line_string._children_:
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
            return list(range(n_points))
        if line_string._kind_ == "linear_ring":
            return [*range(n_points - 1), 0]
        raise ValueError  # never

    @lazy_property
    @staticmethod
    def _line_index_buffer_o_() -> IndexBuffer:
        return IndexBuffer()

    @lazy_property
    @staticmethod
    def _line_index_buffer_(
        line_index_buffer_o: IndexBuffer,
        multi_line_string: MultiLineString3D
    ) -> IndexBuffer:
        return line_index_buffer_o.write(
            StrokeMobject._lump_index_from_getter(StrokeMobject._line_index_getter, multi_line_string)
        )

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

    @lazy_property
    @staticmethod
    def _join_index_buffer_o_() -> IndexBuffer:
        return IndexBuffer()

    @lazy_property
    @staticmethod
    def _join_index_buffer_(
        join_index_buffer_o: IndexBuffer,
        multi_line_string: MultiLineString3D
    ) -> IndexBuffer:
        return join_index_buffer_o.write(
            StrokeMobject._lump_index_from_getter(StrokeMobject._join_index_getter, multi_line_string)
        )

    @classmethod
    def _cap_index_getter(cls, line_string: LineString3D) -> list[int]:
        if line_string._kind_ in "point":
            return []
        n_points = len(line_string._coords_)
        if line_string._kind_ == "line_string":
            # (0, 1, 2, 1, 2, 3, ..., n-3, n-2, n-1)
            return [0, 1, n_points - 2, n_points - 1]
        if line_string._kind_ == "linear_ring":
            return []
        raise ValueError  # never

    @lazy_property
    @staticmethod
    def _cap_index_buffer_o_() -> IndexBuffer:
        return IndexBuffer()

    @lazy_property
    @staticmethod
    def _cap_index_buffer_(
        cap_index_buffer_o: IndexBuffer,
        multi_line_string: MultiLineString3D
    ) -> IndexBuffer:
        return cap_index_buffer_o.write(
            StrokeMobject._lump_index_from_getter(StrokeMobject._cap_index_getter, multi_line_string)
        )

    @classmethod
    def _point_index_getter(cls, line_string: LineString3D) -> list[int]:
        if line_string._kind_ in "point":
            return [0]
        if line_string._kind_ == "line_string":
            return []
        if line_string._kind_ == "linear_ring":
            return []
        raise ValueError  # never

    @lazy_property
    @staticmethod
    def _point_index_buffer_o_() -> IndexBuffer:
        return IndexBuffer()

    @lazy_property
    @staticmethod
    def _point_index_buffer_(
        point_index_buffer_o: IndexBuffer,
        multi_line_string: MultiLineString3D
    ) -> IndexBuffer:
        return point_index_buffer_o.write(
            StrokeMobject._lump_index_from_getter(StrokeMobject._point_index_getter, multi_line_string)
        )

    @lazy_property_writable
    @staticmethod
    def _render_samples_() -> int:
        return 4

    @lazy_property
    @staticmethod
    def _stroke_render_items_(
        single_sided: bool,
        has_linecap: bool,
        line_index_buffer: IndexBuffer,
        join_index_buffer: IndexBuffer,
        cap_index_buffer: IndexBuffer,
        point_index_buffer: IndexBuffer
    ) -> list[tuple[list[str], IndexBuffer, int]]:
        subroutine_name = "single_sided" if single_sided else "both_sided"
        result: list[tuple[list[str], IndexBuffer, int]] = [
            ([
                "#define STROKE_LINE",
                f"#define line_subroutine {subroutine_name}"
            ], line_index_buffer, moderngl.LINE_STRIP),
            ([
                "#define STROKE_JOIN",
                f"#define join_subroutine {subroutine_name}"
            ], join_index_buffer, moderngl.TRIANGLES)
        ]
        if has_linecap and not single_sided:
            result.extend([
                ([
                    "#define STROKE_CAP"
                ], cap_index_buffer, moderngl.LINE_STRIP),
                ([
                    "#define STROKE_POINT"
                ], point_index_buffer, moderngl.POINTS)
            ])
        return result

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        # TODO: Is this already the best practice?
        uniform_blocks = [
            scene_config._camera._ub_camera_,
            self._ub_model_,
            self._ub_stroke_,
            self._ub_winding_sign_o_.write({
                "u_winding_sign": np.array(self._calculate_winding_sign(scene_config._camera))
            })
        ]
        # Render color
        target_framebuffer.depth_mask = False
        for custom_macros, index_buffer, mode in self._stroke_render_items_:
            RenderProcedure.render_step(
                shader_str=RenderProcedure.read_shader("stroke"),
                custom_macros=custom_macros,
                texture_storages=[],
                uniform_blocks=uniform_blocks,
                attributes=self._attributes_,
                index_buffer=index_buffer,
                framebuffer=target_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.BLEND,
                    blend_func=moderngl.ADDITIVE_BLENDING,
                    blend_equation=moderngl.MAX
                ),
                mode=mode
            )
        target_framebuffer.depth_mask = True
        # Render depth
        target_framebuffer.color_mask = (False, False, False, False)
        for custom_macros, index_buffer, mode in self._stroke_render_items_:
            RenderProcedure.render_step(
                shader_str=RenderProcedure.read_shader("stroke"),
                custom_macros=custom_macros,
                texture_storages=[],
                uniform_blocks=uniform_blocks,
                attributes=self._attributes_,
                index_buffer=index_buffer,
                framebuffer=target_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.DEPTH_TEST
                ),
                mode=mode
            )
        target_framebuffer.color_mask = (True, True, True, True)

    def _calculate_winding_sign(self, camera: Camera) -> float:
        # TODO: The calculation here is somehow redundant with what shader does...
        area = 0.0
        transform = camera._projection_matrix_ @ camera._view_matrix_ @ self._model_matrix_
        for line_string in self._multi_line_string_._children_:
            coords_2d = self.apply_affine(transform, line_string._coords_)[:, :2]
            area += np.cross(coords_2d, np.roll(coords_2d, -1, axis=0)).sum()
        return 1.0 if area >= 0.0 else -1.0

    def _get_local_sample_points(self) -> Vec3sT:
        line_strings = self._multi_line_string_._children_
        if not line_strings:
            return np.zeros((0, 3))
        return np.concatenate([
            line_string._coords_
            for line_string in line_strings
        ])

    def _set_style_locally(
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
        if width is not None:
            self._width_ = width
        if single_sided is not None:
            self._single_sided_ = single_sided
        if has_linecap is not None:
            self._has_linecap_ = has_linecap
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        if color_component is not None:
            self._color_ = color_component
        if opacity_component is not None:
            self._opacity_ = opacity_component
        if dilate is not None:
            self._dilate_ = dilate
        if apply_oit is not None:
            self._apply_oit_ = apply_oit
        else:
            if any(param is not None for param in (
                opacity_component,
                dilate
            )):
                self._apply_oit_ = True
        return self

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
        for mobject in self.iter_descendants(broadcast=broadcast):
            if not isinstance(mobject, StrokeMobject):
                continue
            mobject._set_style_locally(
                width=width,
                single_sided=single_sided,
                has_linecap=has_linecap,
                color=color,
                opacity=opacity,
                dilate=dilate,
                apply_oit=apply_oit
            )
        return self
