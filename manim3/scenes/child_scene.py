__all__ = ["ChildScene"]


import moderngl
import numpy as np

from ..custom_typing import (
    ColorType,
    Real,
    Vec3T
)
from ..mobjects.mobject import Mobject
from ..utils.lazy import lazy_property
from ..utils.render_procedure import (
    RenderProcedure,
    TextureStorage
)
from ..utils.renderable import Renderable
from ..utils.scene_config import SceneConfig


class ChildScene(Renderable):
    def __init__(self):
        self._mobject_node: Mobject = Mobject()
        self._scene_config: SceneConfig = SceneConfig()

    def _update_dt(self, dt: Real):
        for mobject in self._mobject_node.iter_descendants():
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

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        # Inspired from https://github.com/ambrosiogabe/MathAnimation
        # ./Animations/src/renderer/Renderer.cpp
        opaque_mobjects: list[Mobject] = []
        transparent_mobjects: list[Mobject] = []
        for mobject in self._mobject_node.iter_descendants():
            if not mobject._has_local_sample_points():
                continue
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
                    RenderProcedure.fullscreen_render_step(
                        shader_str=RenderProcedure.read_shader("copy"),
                        custom_macros=[
                            "#define COPY_DEPTH"
                        ],
                        texture_storages=[
                            self._u_color_map_o_.write(
                                np.array(component_texture)
                            ),
                            self._u_depth_map_o_.write(
                                np.array(component_depth_texture)
                            )
                        ],
                        uniform_blocks=[],
                        framebuffer=opaque_framebuffer,
                        context_state=RenderProcedure.context_state(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=(moderngl.ONE, moderngl.ZERO)
                        )
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
                    RenderProcedure.fullscreen_render_step(
                        shader_str=RenderProcedure.read_shader("oit_accum"),
                        custom_macros=[],
                        texture_storages=[
                            u_color_map,
                            u_depth_map
                        ],
                        uniform_blocks=[],
                        framebuffer=accum_framebuffer,
                        context_state=RenderProcedure.context_state(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=moderngl.ADDITIVE_BLENDING
                        )
                    )
                    RenderProcedure.fullscreen_render_step(
                        shader_str=RenderProcedure.read_shader("oit_revealage"),
                        custom_macros=[],
                        texture_storages=[
                            u_color_map,
                            u_depth_map
                        ],
                        uniform_blocks=[],
                        framebuffer=revealage_framebuffer,
                        context_state=RenderProcedure.context_state(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=(moderngl.ZERO, moderngl.ONE_MINUS_SRC_COLOR)
                        )
                    )

            RenderProcedure.fullscreen_render_step(
                shader_str=RenderProcedure.read_shader("copy"),
                custom_macros=[
                    "#define COPY_DEPTH"
                ],
                texture_storages=[
                    self._u_color_map_o_.write(
                        np.array(opaque_texture)
                    ),
                    self._u_depth_map_o_.write(
                        np.array(depth_texture)
                    )
                ],
                uniform_blocks=[],
                framebuffer=target_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                    blend_func=(moderngl.ONE, moderngl.ZERO)
                )
            )
            RenderProcedure.fullscreen_render_step(
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
                framebuffer=target_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.BLEND | moderngl.DEPTH_TEST
                )
            )

    def add(self, *mobjects: Mobject):
        self._mobject_node.add(*mobjects)
        return self

    def remove(self, *mobjects: Mobject):
        self._mobject_node.remove(*mobjects)
        return self

    def set_view(
        self,
        *,
        eye: Vec3T | None = None,
        target: Vec3T | None = None,
        up: Vec3T | None = None
    ):
        self._scene_config.set_view(
            eye=eye,
            target=target,
            up=up
        )

    def set_background(
        self,
        *,
        color: ColorType | None = None,
        opacity: Real | None = None
    ):
        self._scene_config.set_background(
            color=color,
            opacity=opacity
        )
        return self

    def set_ambient_light(
        self,
        *,
        color: ColorType | None = None,
        opacity: Real | None = None
    ):
        self._scene_config.set_ambient_light(
            color=color,
            opacity=opacity
        )
        return self

    def add_point_light(
        self,
        *,
        position: Vec3T | None = None,
        color: ColorType | None = None,
        opacity: Real | None = None
    ):
        self._scene_config.add_point_light(
            position=position,
            color=color,
            opacity=opacity
        )
        return self

    def set_point_light(
        self,
        *,
        index: int | None = None,
        position: Vec3T | None = None,
        color: ColorType | None = None,
        opacity: Real | None = None
    ):
        self._scene_config.set_point_light(
            index=index,
            position=position,
            color=color,
            opacity=opacity
        )
        return self

    def set_style(
        self,
        *,
        background_color: ColorType | None = None,
        background_opacity: Real | None = None,
        ambient_light_color: ColorType | None = None,
        ambient_light_opacity: Real | None = None,
        point_light_position: Vec3T | None = None,
        point_light_color: ColorType | None = None,
        point_light_opacity: Real | None = None
    ):
        self._scene_config.set_style(
            background_color=background_color,
            background_opacity=background_opacity,
            ambient_light_color=ambient_light_color,
            ambient_light_opacity=ambient_light_opacity,
            point_light_position=point_light_position,
            point_light_color=point_light_color,
            point_light_opacity=point_light_opacity
        )
        return self
