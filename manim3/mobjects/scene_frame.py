#import time

import numpy as np
from PIL import Image

from ..lazy.core import LazyDynamicContainer
from ..lazy.interface import Lazy
from ..mobjects.mobject import Mobject
from ..passes.render_pass import RenderPass
from ..rendering.config import ConfigSingleton
from ..rendering.context import (
    Context,
    ContextState
)
from ..rendering.framebuffer import (
    ColorFramebuffer,
    Framebuffer,
    OpaqueFramebuffer,
    TransparentFramebuffer
)
from ..rendering.gl_buffer import TextureIDBuffer
from ..rendering.mgl_enums import ContextFlag
from ..rendering.texture import TextureFactory
from ..rendering.vertex_array import VertexArray


class SceneFrame(Mobject):
    __slots__ = ()

    @Lazy.variable_collection
    @classmethod
    def _render_passes_(cls) -> list[RenderPass]:
        return []

    @Lazy.property
    @classmethod
    def _pixelated_vertex_array_(cls) -> VertexArray:
        return VertexArray(
            shader_filename="copy",
            texture_id_buffers=[
                TextureIDBuffer(
                    field="sampler2D t_color_map"
                )
            ]
        )

    @Lazy.property
    @classmethod
    def _oit_compose_vertex_array_(cls) -> VertexArray:
        return VertexArray(
            shader_filename="oit_compose",
            texture_id_buffers=[
                TextureIDBuffer(
                    field="sampler2D t_accum_map"
                ),
                TextureIDBuffer(
                    field="sampler2D t_revealage_map"
                )
            ]
        )

    def _render_scene(
        self,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        # Inspired from `https://github.com/ambrosiogabe/MathAnimation`
        # `./Animations/src/renderer/Renderer.cpp`.
        opaque_mobjects: list[Mobject] = []
        transparent_mobjects: list[Mobject] = []
        for mobject in self.iter_descendants():
            if not mobject._has_local_sample_points_.value:
                continue
            mobject._scene_state_ = self._scene_state_
            if mobject._is_transparent_.value:
                transparent_mobjects.append(mobject)
            else:
                opaque_mobjects.append(mobject)

        with TextureFactory.depth_texture() as depth_texture:
            opaque_framebuffer = OpaqueFramebuffer(
                color_texture=target_framebuffer.color_texture,
                depth_texture=depth_texture
            )
            opaque_framebuffer.framebuffer.clear()
            for mobject in opaque_mobjects:
                mobject._render(opaque_framebuffer)

            with TextureFactory.texture(dtype="f2") as accum_texture, \
                    TextureFactory.texture(components=1) as revealage_texture:
                transparent_framebuffer = TransparentFramebuffer(
                    accum_texture=accum_texture,
                    revealage_texture=revealage_texture,
                    depth_texture=depth_texture
                )
                # Test against each fragment by the depth buffer, but never write to it.
                transparent_framebuffer.framebuffer.depth_mask = False
                transparent_framebuffer.framebuffer.clear()
                # Initialize `revealage` with 1.0.
                # TODO: There should be a more elegant way using `clear`.
                revealage_texture.write(b"\xff" * (revealage_texture.width * revealage_texture.height))
                for mobject in transparent_mobjects:
                    mobject._render(transparent_framebuffer)

            self._oit_compose_vertex_array_.render(
                framebuffer=target_framebuffer,
                texture_array_dict={
                    "t_accum_map": np.array(accum_texture),
                    "t_revealage_map": np.array(revealage_texture)
                },
                context_state=ContextState(
                    flags=(ContextFlag.BLEND,)
                )
            )

    def _render_scene_with_passes(
        self,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        render_passes = self._render_passes_
        if not render_passes:
            self._render_scene(target_framebuffer)
            return

        with TextureFactory.texture() as teture_0, TextureFactory.texture() as teture_1:
            framebuffers = (
                ColorFramebuffer(
                    color_texture=teture_0
                ),
                ColorFramebuffer(
                    color_texture=teture_1
                )
            )
            target_id = 0
            self._render_scene(framebuffers[0])
            for render_pass in render_passes[:-1]:
                target_id = 1 - target_id
                render_pass._render(
                    texture=framebuffers[1 - target_id].color_texture,
                    target_framebuffer=framebuffers[target_id]
                )
            render_passes[-1]._render(
                texture=framebuffers[target_id].color_texture,
                target_framebuffer=target_framebuffer
            )

    def _process_rendering(
        self,
        *,
        render_to_video: bool = False,
        render_to_image: bool = False
    ) -> None:
        scene_state = self._scene_state_
        red, green, blue = scene_state._background_color_.value
        alpha = scene_state._background_opacity_.value
        with TextureFactory.texture() as color_texture:
            framebuffer = ColorFramebuffer(
                color_texture=color_texture
            )
            framebuffer.framebuffer.clear(red=red, green=green, blue=blue, alpha=alpha)
            self._render_scene_with_passes(framebuffer)

            if render_to_video:
                if ConfigSingleton().rendering.write_video:
                    writing_process = Context.writing_process
                    assert writing_process.stdin is not None
                    writing_process.stdin.write(framebuffer.framebuffer.read(components=4))
                if ConfigSingleton().rendering.preview:
                    window = Context.window
                    if window.is_closing:
                        raise KeyboardInterrupt
                    window.clear()
                    self._pixelated_vertex_array_.render(
                        framebuffer=Framebuffer(
                            framebuffer=Context.window_framebuffer,
                            default_context_state=ContextState(
                                flags=()
                            )
                        ),
                        texture_array_dict={
                            "t_color_map": np.array(color_texture)
                        }
                    )
                    #if (previous_timestamp := self._previous_frame_rendering_timestamp) is not None and \
                    #        (sleep_t := (1.0 / ConfigSingleton().rendering.fps) - (time.time() - previous_timestamp)) > 0.0:
                    #    time.sleep(sleep_t)
                    window.swap_buffers()
                #self._previous_frame_rendering_timestamp = time.time()

            if render_to_image:
                scene_name = ConfigSingleton().rendering.scene_name
                image = Image.frombytes(
                    "RGBA",
                    ConfigSingleton().size.pixel_size,
                    framebuffer.framebuffer.read(components=4),
                    "raw"
                ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                image.save(ConfigSingleton().path.output_dir.joinpath(f"{scene_name}.png"))

    #def add_pass(
    #    self,
    #    *render_passes: RenderPass
    #):
    #    self._render_passes_.extend(render_passes)
    #    return self

    #def discard_pass(
    #    self,
    #    *render_passes: RenderPass
    #):
    #    self._render_passes_.discard(*render_passes)
    #    return self

    @property
    def render_passes(self) -> LazyDynamicContainer[RenderPass]:
        return self._render_passes_
