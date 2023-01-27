__all__ = [
    "ChildScene",
    "Scene"
]


import time

import moderngl
import numpy as np

from ..custom_typing import Real
from ..mobjects.mobject import Mobject
from ..utils.lazy import lazy_property
from ..utils.render_procedure import (
    AttributesBuffer,
    IndexBuffer,
    RenderProcedure,
    TextureStorage
)
from ..utils.scene_config import SceneConfig


class ChildScene(Mobject):
    @lazy_property
    @staticmethod
    def _scene_config_() -> SceneConfig:
        return SceneConfig()

    def _update_dt(self, dt: Real):
        for mobject in self.get_descendants_excluding_self():
            mobject._update_dt(dt)
        return self

    @lazy_property
    @staticmethod
    def _u_color_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_color_map")

    @lazy_property
    @staticmethod
    def _u_accum_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_accum_map")

    @lazy_property
    @staticmethod
    def _u_revealage_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_revealage_map")

    @lazy_property
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

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        # Inspired from https://github.com/ambrosiogabe/MathAnimation
        # ./Animations/src/renderer/Renderer.cpp
        opaque_mobjects: list[Mobject] = []
        transparent_mobjects: list[Mobject] = []
        for mobject in self.get_descendants_excluding_self():
            if mobject._apply_oit_:
                transparent_mobjects.append(mobject)
            else:
                opaque_mobjects.append(mobject)

        with RenderProcedure.texture() as opaque_texture, \
                RenderProcedure.texture(dtype="f2") as accum_texture, \
                RenderProcedure.texture(components=1) as revealage_texture, \
                RenderProcedure.depth_texture() as depth_texture, \
                RenderProcedure.framebuffer(
                    color_attachments=[opaque_texture],
                    depth_attachment=depth_texture
                ) as opaque_framebuffer, \
                RenderProcedure.framebuffer(
                    color_attachments=[accum_texture],
                    depth_attachment=depth_texture
                ) as accum_framebuffer, \
                RenderProcedure.framebuffer(
                    color_attachments=[revealage_texture],
                    depth_attachment=depth_texture
                ) as revealage_framebuffer:

            for mobject in opaque_mobjects:
                with RenderProcedure.texture() as component_texture, \
                        RenderProcedure.depth_texture() as component_depth_texture, \
                        RenderProcedure.framebuffer(
                            color_attachments=[component_texture],
                            depth_attachment=component_depth_texture
                        ) as component_framebuffer:
                    mobject._render_with_passes(scene_config, component_framebuffer)
                    RenderProcedure.render_step(
                        shader_str=RenderProcedure.read_shader("copy"),
                        custom_macros=[],
                        texture_storages=[
                            self._u_color_map_o_.write(
                                np.array(component_texture)
                            ),
                            self._u_depth_map_o_.write(
                                np.array(component_depth_texture)
                            )
                        ],
                        uniform_blocks=[],
                        attributes=self._attributes_,
                        index_buffer=self._index_buffer_,
                        framebuffer=opaque_framebuffer,
                        context_state=RenderProcedure.context_state(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=(moderngl.ONE, moderngl.ZERO)
                        ),
                        mode=moderngl.TRIANGLE_FAN
                    )

            # Test against each fragment by the depth buffer, but never write to it.
            accum_framebuffer.depth_mask = False
            revealage_framebuffer.depth_mask = False
            revealage_framebuffer.clear(red=1.0)  # initialize `revealage` with 1.0
            for mobject in transparent_mobjects:
                with RenderProcedure.texture() as component_texture, \
                        RenderProcedure.depth_texture() as component_depth_texture, \
                        RenderProcedure.framebuffer(
                            color_attachments=[component_texture],
                            depth_attachment=component_depth_texture
                        ) as component_framebuffer:
                    mobject._render_with_passes(scene_config, component_framebuffer)
                    u_color_map = self._u_color_map_o_.write(
                        np.array(component_texture)
                    )
                    u_depth_map = self._u_depth_map_o_.write(
                        np.array(component_depth_texture)
                    )
                    RenderProcedure.render_step(
                        shader_str=RenderProcedure.read_shader("oit_accum"),
                        custom_macros=[],
                        texture_storages=[
                            u_color_map,
                            u_depth_map
                        ],
                        uniform_blocks=[],
                        attributes=self._attributes_,
                        index_buffer=self._index_buffer_,
                        framebuffer=accum_framebuffer,
                        context_state=RenderProcedure.context_state(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=moderngl.ADDITIVE_BLENDING
                        ),
                        mode=moderngl.TRIANGLE_FAN
                    )
                    RenderProcedure.render_step(
                        shader_str=RenderProcedure.read_shader("oit_revealage"),
                        custom_macros=[],
                        texture_storages=[
                            u_color_map,
                            u_depth_map
                        ],
                        uniform_blocks=[],
                        attributes=self._attributes_,
                        index_buffer=self._index_buffer_,
                        framebuffer=revealage_framebuffer,
                        context_state=RenderProcedure.context_state(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=(moderngl.ZERO, moderngl.ONE_MINUS_SRC_COLOR)
                        ),
                        mode=moderngl.TRIANGLE_FAN
                    )

            RenderProcedure.render_step(
                shader_str=RenderProcedure.read_shader("copy"),
                custom_macros=[],
                texture_storages=[
                    self._u_color_map_o_.write(
                        np.array(opaque_texture)
                    ),
                    self._u_depth_map_o_.write(
                        np.array(depth_texture)
                    )
                ],
                uniform_blocks=[],
                attributes=self._attributes_,
                index_buffer=self._index_buffer_,
                framebuffer=target_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                    blend_func=(moderngl.ONE, moderngl.ZERO)
                ),
                mode=moderngl.TRIANGLE_FAN
            )
            RenderProcedure.render_step(
                shader_str=RenderProcedure.read_shader("oit_compose"),
                custom_macros=[],
                texture_storages=[
                    self._u_accum_map_o_.write(
                        np.array(accum_texture)
                    ),
                    self._u_revealage_map_o_.write(
                        np.array(revealage_texture)
                    )
                ],
                uniform_blocks=[],
                attributes=self._attributes_,
                index_buffer=self._index_buffer_,
                framebuffer=target_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.BLEND | moderngl.DEPTH_TEST
                ),
                mode=moderngl.TRIANGLE_FAN
            )


class Scene(ChildScene):
    def _render_scene(self) -> None:
        framebuffer = RenderProcedure._WINDOW_FRAMEBUFFER
        framebuffer.clear()
        self._render_with_passes(self._scene_config_, framebuffer)

    def wait(self, t: Real):
        window = RenderProcedure._WINDOW
        #if window is None:
        #    return self  # TODO
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
