__all__ = ["CopyPass"]


import moderngl
import numpy as np

from manim3.utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)

from ..render_passes.render_pass import RenderPass
from ..utils.renderable import (
    AttributesBuffer,
    ContextState,
    Framebuffer,
    IndexBuffer,
    IntermediateFramebuffer,
    RenderStep,
    TextureStorage
)


class CopyPass(RenderPass):
    @lazy_property_initializer
    @staticmethod
    def _u_color_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_color_map")

    @lazy_property_initializer
    @staticmethod
    def _u_depth_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_depth_map")

    @lazy_property
    @staticmethod
    def _attributes_() -> AttributesBuffer:
        return AttributesBuffer([
            "vec3 in_position",
            "vec2 in_uv"
        ]).write({
            "in_position": np.array([
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ]),
            "in_uv": np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ])
        })

    @lazy_property
    @staticmethod
    def _index_buffer_() -> IndexBuffer:
        return IndexBuffer().write(np.array((
            0, 1, 2, 3
        )))

    @lazy_property_initializer_writable
    @staticmethod
    def _enable_only_() -> int:
        return moderngl.NOTHING

    @lazy_property_initializer_writable
    @staticmethod
    def _context_state_() -> ContextState:
        return ContextState()

    def set(
        self,
        *,
        enable_only: int | None = None,
        context_state: ContextState | None = None
    ):
        if enable_only is not None:
            self._enable_only_ = enable_only
        if context_state is not None:
            self._context_state_ = context_state
        return self

    def render(
        self,
        input_framebuffer: IntermediateFramebuffer,
        output_framebuffer: Framebuffer
    ):
        self.render_by_step(RenderStep(
            shader_str=self._read_shader("copy"),
            texture_storages=[
                self._u_color_map_o_.write(
                    np.array(input_framebuffer.get_attachment(0))
                ),
                self._u_depth_map_o_.write(
                    np.array(input_framebuffer.get_attachment(-1))
                )
            ],
            uniform_blocks=[],
            subroutines={},
            attributes=self._attributes_,
            index_buffer=self._index_buffer_,
            framebuffer=output_framebuffer,
            enable_only=self._enable_only_,
            context_state=self._context_state_,
            mode=moderngl.TRIANGLE_FAN
        ))
