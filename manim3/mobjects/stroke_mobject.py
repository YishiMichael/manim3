__all__ = ["StrokeMobject"]


import itertools as it

from colour import Color
import moderngl
import numpy as np

from ..custom_typing import (
    ColorType,
    Real,
    Vec3sT
)
from ..mobjects.mobject import Mobject
from ..mobjects.mesh_mobject import MeshMobject
from ..render_passes.copy_pass import CopyPass
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)
from ..utils.renderable import (
    AttributesBuffer,
    ContextState,
    Framebuffer,
    IndexBuffer,
    IntermediateDepthTextures,
    IntermediateFramebuffer,
    IntermediateTextures,
    RenderStep,
    Renderable,
    UniformBlockBuffer
)
from ..utils.scene_config import SceneConfig


class StrokeMobject(Mobject):
    def __init__(self, position: Vec3sT, is_loop: bool):
        super().__init__()
        self._position_ = position
        self._is_loop_ = is_loop

    @lazy_property_initializer_writable
    @staticmethod
    def _position_() -> Vec3sT:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _is_loop_() -> bool:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _width_() -> Real:
        return 0.04

    @lazy_property_initializer_writable
    @staticmethod
    def _color_() -> ColorType:
        return Color("white")

    @lazy_property_initializer_writable
    @staticmethod
    def _dilate_() -> Real:
        return 0.0

    @lazy_property_initializer_writable
    @staticmethod
    def _single_sided_() -> bool:
        return False

    @lazy_property_initializer
    @staticmethod
    def _ub_stroke_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer("ub_stroke", [
            "float u_stroke_width",
            "vec4 u_stroke_color",
            "float u_stroke_dilate"
        ])

    @lazy_property
    @staticmethod
    def _ub_stroke_(
        ub_stroke_o: UniformBlockBuffer,
        width: Real,
        color: ColorType,
        dilate: Real
    ) -> UniformBlockBuffer:
        ub_stroke_o.write({
            "u_stroke_width": np.array(width),
            "u_stroke_color": MeshMobject._color_to_vector(color),
            "u_stroke_dilate": np.array(dilate)
        })
        return ub_stroke_o

    @lazy_property_initializer
    @staticmethod
    def _attributes_o_() -> AttributesBuffer:
        return AttributesBuffer([
            "vec3 in_position"
        ])

    @lazy_property
    @staticmethod
    def _attributes_(
        attributes_o: AttributesBuffer,
        position: Vec3sT
    ) -> AttributesBuffer:
        attributes_o.write({
            "in_position": position
        })
        return attributes_o

    @lazy_property_initializer
    @staticmethod
    def _line_index_buffer_o_() -> IndexBuffer:
        return IndexBuffer()

    @lazy_property
    @staticmethod
    def _line_index_buffer_(
        line_index_buffer_o: IndexBuffer,
        position: Vec3sT,
        is_loop: bool
    ) -> IndexBuffer:
        index_array = np.arange(len(position))
        if is_loop:
            index_array = np.append(index_array, 0)
        line_index_buffer_o.write(index_array)
        return line_index_buffer_o

    @lazy_property_initializer
    @staticmethod
    def _join_index_buffer_o_() -> IndexBuffer:
        return IndexBuffer()

    @lazy_property
    @staticmethod
    def _join_index_buffer_(
        join_index_buffer_o: IndexBuffer,
        position: Vec3sT,
        is_loop: bool
    ) -> IndexBuffer:
        n = len(position)
        index_array = np.array(list(zip(range(n - 2), range(1, n - 1), range(2, n)))).flatten()
        if is_loop:
            index_array = np.append(index_array, np.array((n - 2, n - 1, 0, n - 1, 0, 1)))
        join_index_buffer_o.write(index_array)
        return join_index_buffer_o

    @lazy_property
    @staticmethod
    def _copy_pass_() -> CopyPass:
        return CopyPass()

    def _render(self, scene_config: SceneConfig, target_framebuffer: Framebuffer) -> None:
        with IntermediateTextures.register_n(1) as textures:
            with IntermediateDepthTextures.register_n(1) as depth_textures:
                intermediate_framebuffer = IntermediateFramebuffer(textures, depth_textures[0])
                self._render_by_step(RenderStep(
                    shader_str=Renderable._read_shader("stroke_line"),
                    texture_storages=[],
                    uniform_blocks=[
                        scene_config._camera_._ub_camera_,
                        self._ub_model_,
                        self._ub_stroke_
                    ],
                    attributes=self._attributes_,
                    subroutines={
                        "line_dilate_func": "single_sided_dilate" if self._single_sided_ else "both_sided_dilate"
                    },
                    index_buffer=self._line_index_buffer_,
                    framebuffer=intermediate_framebuffer,
                    enable_only=moderngl.BLEND,
                    context_state=ContextState(
                        blend_func=moderngl.ADDITIVE_BLENDING,
                        blend_equation=moderngl.MAX
                    ),
                    mode=moderngl.LINE_STRIP
                ), RenderStep(
                    shader_str=Renderable._read_shader("stroke_join"),
                    texture_storages=[],
                    uniform_blocks=[
                        scene_config._camera_._ub_camera_,
                        self._ub_model_,
                        self._ub_stroke_
                    ],
                    attributes=self._attributes_,
                    subroutines={
                        "join_dilate_func": "single_sided_dilate" if self._single_sided_ else "both_sided_dilate"
                    },
                    index_buffer=self._join_index_buffer_,
                    framebuffer=intermediate_framebuffer,
                    enable_only=moderngl.BLEND,
                    context_state=ContextState(
                        blend_func=moderngl.ADDITIVE_BLENDING,
                        blend_equation=moderngl.MAX
                    ),
                    mode=moderngl.TRIANGLES
                ))
                self._copy_pass_._render(
                    input_framebuffer=intermediate_framebuffer,
                    output_framebuffer=target_framebuffer,
                    mobject=self,
                    scene_config=scene_config
                )
