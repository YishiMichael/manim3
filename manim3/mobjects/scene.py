__all__ = ["Scene"]


import time

import moderngl
from moderngl_window.context.pyglet.window import Window as PygletWindow
import numpy as np

from ..constants import (
    PIXEL_HEIGHT,
    PIXEL_WIDTH
)
from ..custom_typing import Real
from ..mobjects.mobject import Mobject
from ..render_passes.copy_pass import CopyPass
from ..utils.context_singleton import ContextSingleton
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer
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
    TextureStorage
)
from ..utils.scene_config import SceneConfig


class Scene(Mobject):
    def __init__(self, *, main: bool = False):
        if main:
            window = ContextSingleton.get_window()
            framebuffer = Framebuffer(
                ContextSingleton().detect_framebuffer()
            )
        else:
            window = None
            framebuffer = IntermediateFramebuffer(
                color_attachments=[
                    ContextSingleton().texture(
                        size=(PIXEL_WIDTH, PIXEL_HEIGHT),
                        components=4
                    )
                ],
                depth_attachment=ContextSingleton().depth_texture(
                    size=(PIXEL_WIDTH, PIXEL_HEIGHT)
                )
            )

        self._window: PygletWindow | None = window
        self._framebuffer: Framebuffer = framebuffer
        super().__init__()

    @lazy_property_initializer
    @staticmethod
    def _scene_config_() -> SceneConfig:
        return SceneConfig()

    @lazy_property_initializer
    @staticmethod
    def _u_color_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_color_map")

    @lazy_property_initializer
    @staticmethod
    def _u_accum_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_accum_map")

    @lazy_property_initializer
    @staticmethod
    def _u_revealage_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_revealage_map")

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

    def _render(self, scene_config: SceneConfig, target_framebuffer: Framebuffer) -> None:
        # Inspired from https://github.com/ambrosiogabe/MathAnimation
        # ./Animations/src/renderer/Renderer.cpp
        opaque_mobjects: list[Mobject] = []
        transparent_mobjects: list[Mobject] = []
        for mobject in self.get_descendants_excluding_self():
            if mobject._apply_oit_:
                transparent_mobjects.append(mobject)
            else:
                opaque_mobjects.append(mobject)

        component_texture = IntermediateTextures.fetch()
        opaque_texture = IntermediateTextures.fetch()
        accum_texture = IntermediateTextures.fetch(dtype="f2")
        revealage_texture = IntermediateTextures.fetch(components=1)
        component_depth_texture = IntermediateDepthTextures.fetch()
        depth_texture = IntermediateDepthTextures.fetch()
        component_target_framebuffer = IntermediateFramebuffer([component_texture], component_depth_texture)

        opaque_target_framebuffer = IntermediateFramebuffer([opaque_texture], depth_texture)
        opaque_target_framebuffer._framebuffer.depth_mask = True
        opaque_target_framebuffer.clear()
        for mobject in opaque_mobjects:
            component_target_framebuffer._framebuffer.depth_mask = True
            component_target_framebuffer.clear()
            mobject._render_with_passes(scene_config, component_target_framebuffer)
            #vals = np.frombuffer(depth_texture.read(), dtype=np.float32)
            #print(vals.min(), vals.max())
            CopyPass(
                enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                context_state=ContextState(
                    blend_func=(moderngl.ONE, moderngl.ZERO)
                )
            )._render(
                input_framebuffer=component_target_framebuffer,
                output_framebuffer=opaque_target_framebuffer
            )
            #vals = np.frombuffer(depth_texture.read(), dtype=np.float32)
            #print(vals.min(), vals.max())
            #print()
        #print("####")

        # Test against each fragment by the depth buffer, but never write to it.
        # We should prevent from clearing buffer bits.
        accum_target_framebuffer = IntermediateFramebuffer([accum_texture], depth_texture)
        accum_target_framebuffer._framebuffer.depth_mask = False
        accum_target_framebuffer.clear()
        revealage_target_framebuffer = IntermediateFramebuffer([revealage_texture], depth_texture)
        revealage_target_framebuffer._framebuffer.depth_mask = False
        revealage_target_framebuffer.clear(red=1.0)  # initialize `revealage` with 1.0
        for mobject in transparent_mobjects:
            component_target_framebuffer._framebuffer.depth_mask = True
            component_target_framebuffer.clear()
            mobject._render_with_passes(scene_config, component_target_framebuffer)
            u_color_map = self._u_color_map_o_.write(
                np.array(component_texture)
            )
            u_depth_map = self._u_depth_map_o_.write(
                np.array(component_depth_texture)
            )
            #vals = np.frombuffer(depth_texture.read(), dtype=np.float32)
            #print(vals.min(), vals.max(), np.average(vals))
            self._render_by_step(RenderStep(
                shader_str=Renderable._read_shader("oit_accum"),
                texture_storages=[
                    u_color_map,
                    u_depth_map
                ],
                uniform_blocks=[],
                subroutines={},
                attributes=self._attributes_,
                index_buffer=self._index_buffer_,
                framebuffer=accum_target_framebuffer,
                enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                context_state=ContextState(
                    blend_func=moderngl.ADDITIVE_BLENDING
                ),
                mode=moderngl.TRIANGLE_FAN
            ), RenderStep(
                shader_str=Renderable._read_shader("oit_revealage"),
                texture_storages=[
                    u_color_map,
                    u_depth_map
                ],
                uniform_blocks=[],
                subroutines={},
                attributes=self._attributes_,
                index_buffer=self._index_buffer_,
                framebuffer=revealage_target_framebuffer,
                enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                context_state=ContextState(
                    blend_func=(moderngl.ZERO, moderngl.ONE_MINUS_SRC_COLOR)
                ),
                mode=moderngl.TRIANGLE_FAN
            ))
            #vals = np.frombuffer(depth_texture.read(), dtype=np.float32)
            #print(vals.min(), vals.max(), np.average(vals))
            #print()
        #print("####")
        #from PIL import Image
        #Image.frombytes('RGB', accum_target_framebuffer._framebuffer.size, accum_target_framebuffer._framebuffer.read(), 'raw').show()

        CopyPass(
            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
            context_state=ContextState()
        )._render(
            input_framebuffer=opaque_target_framebuffer,
            output_framebuffer=target_framebuffer
        )
        self._render_by_step(RenderStep(
            shader_str=Renderable._read_shader("oit_compose"),
            texture_storages=[
                self._u_accum_map_o_.write(
                    np.array(accum_texture)
                ),
                self._u_revealage_map_o_.write(
                    np.array(revealage_texture)
                )
            ],
            uniform_blocks=[],
            subroutines={},
            attributes=self._attributes_,
            index_buffer=self._index_buffer_,
            framebuffer=target_framebuffer,
            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
            context_state=ContextState(),
            mode=moderngl.TRIANGLE_FAN
        ))

        IntermediateTextures.restore(component_texture)
        IntermediateTextures.restore(opaque_texture)
        IntermediateTextures.restore(accum_texture)
        IntermediateTextures.restore(revealage_texture)
        IntermediateDepthTextures.restore(component_depth_texture)
        IntermediateDepthTextures.restore(depth_texture)
        component_target_framebuffer.release()
        opaque_target_framebuffer.release()
        accum_target_framebuffer.release()
        revealage_target_framebuffer.release()

    def _render_scene(self) -> None:
        framebuffer = self._framebuffer
        framebuffer.clear()
        self._render_with_passes(self._scene_config_, framebuffer)

    def _update_dt(self, dt: Real):
        for mobject in self.get_descendants_excluding_self():
            mobject._update_dt(dt)
        return self

    def wait(self, t: Real):
        window = self._window
        if window is None:
            return self  # TODO

        FPS = 30.0
        dt = 1.0 / FPS
        elapsed_time = 0.0
        timestamp = time.time()
        while not window.is_closing and elapsed_time < t:
            elapsed_time += dt
            delta_t = time.time() - timestamp
            if dt > delta_t:
                time.sleep(dt - delta_t)
            timestamp = time.time()
            window.clear()
            self._update_dt(dt)
            self._render_scene()
            window.swap_buffers()
        return self
