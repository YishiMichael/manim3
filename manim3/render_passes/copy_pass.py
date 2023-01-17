__all__ = ["CopyPass"]


import moderngl
import numpy as np

from manim3.utils.lazy import (
    lazy_property,
    lazy_property_initializer
)

from ..mobjects.mobject import Mobject, SceneConfig
from ..render_passes.render_pass import RenderPass
from ..utils.renderable import (
    AttributesBuffer,
    ContextState,
    Framebuffer,
    IndexBuffer,
    IntermediateFramebuffer,
    RenderStep,
    Renderable,
    TextureStorage
)


class CopyPass(RenderPass[Mobject]):
    _INSTANCE: "CopyPass | None" = None

    def __new__(cls) -> "CopyPass":
        if cls._INSTANCE is None:
            cls._INSTANCE = super().__new__(cls)
        return cls._INSTANCE

    @lazy_property_initializer
    @staticmethod
    def _u_color_maps_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_color_maps[NUM_U_COLOR_MAPS]")

    @lazy_property_initializer
    @staticmethod
    def _u_depth_maps_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_depth_maps[NUM_U_DEPTH_MAPS]")

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

    def _render(
        self,
        input_framebuffer: IntermediateFramebuffer,
        output_framebuffer: Framebuffer,
        mobject: Mobject,
        scene_config: SceneConfig
    ):
        self._render_by_step(RenderStep(
            shader_str=Renderable._read_shader("copy"),
            texture_storages=[
                self._u_color_maps_o_.write(
                    np.array(input_framebuffer._color_attachments)
                ),
                self._u_depth_maps_o_.write(
                    np.array(
                        [depth_attachment]
                        if (depth_attachment := input_framebuffer._depth_attachment) is not None
                        else []
                    )
                )
            ],
            uniform_blocks=[],
            attributes=self._attributes_,
            subroutines={},
            index_buffer=self._index_buffer_,
            framebuffer=output_framebuffer,
            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,  # TODO
            context_state=ContextState(),  # TODO
            mode=moderngl.TRIANGLE_FAN
        ))
