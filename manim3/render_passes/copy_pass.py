__all__ = ["CopyPass"]


import moderngl
import numpy as np

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
                TextureStorage("sampler2D u_color_maps[NUM_U_COLOR_MAPS]").write(
                    np.array(input_framebuffer._color_attachments)
                ),
                TextureStorage("sampler2D u_depth_maps[NUM_U_DEPTH_MAPS]").write(
                    np.array(
                        [depth_attachment]
                        if (depth_attachment := input_framebuffer._depth_attachment) is not None
                        else []
                    )
                )
            ],
            uniform_blocks=[],
            attributes=AttributesBuffer([
                "vec3 a_position",
                "vec2 a_uv"
            ]).write({
                "a_position": np.array([
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                ]),
                "a_uv": np.array([
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ])
            }),
            subroutines={},
            index_buffer=IndexBuffer().write(np.array([
                0, 1, 3, 1, 2, 3
            ])),
            framebuffer=output_framebuffer,
            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,  # TODO
            context_state=ContextState(),  # TODO
            mode=moderngl.TRIANGLES
        ))
