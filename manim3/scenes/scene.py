__all__ = ["Scene"]


import time
import warnings

import moderngl
import numpy as np

from ..animations.animation import Animation
from ..custom_typing import (
    ColorType,
    Real,
    Vec3T
)
from ..mobjects.mobject import Mobject
from ..scenes.active_scene_data import ActiveSceneDataSingleton
from ..scenes.scene_config import SceneConfig
from ..rendering.config import ConfigSingleton
from ..rendering.render_procedure import (
    ContextSingleton,
    RenderProcedure,
    TextureStorage
)
#from ..rendering.renderable import Renderable
from ..utils.lazy import (
    LazyData,
    lazy_basedata,
    lazy_property
)


class Scene(Mobject):
    #def __init__(self):
    #    self._scene_config: SceneConfig = SceneConfig()
    #    #self._mobject_node: Mobject = Mobject()
    #    self._animations: dict[Animation, float] = {}
    #    self._frame_floating_index: float = 0.0  # A timer scaled by fps
    #    self._previous_rendering_timestamp: float | None = None

    @lazy_basedata
    @staticmethod
    def _scene_config_() -> SceneConfig:
        return SceneConfig()

    @lazy_basedata
    @staticmethod
    def _animation_dict_() -> dict[Animation, float]:
        return {}

    @lazy_basedata
    @staticmethod
    def _frame_floating_index_() -> float:
        # A timer scaled by fps
        return 0.0

    @lazy_basedata
    @staticmethod
    def _previous_rendering_timestamp_() -> float | None:
        return None

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
        for mobject in self.iter_descendants():
            if not mobject._has_local_sample_points_:
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

    def _render_frame(self) -> None:
        scene_config = self._scene_config_
        red, green, blue = scene_config._background_color_
        alpha = scene_config._background_opacity_

        active_scene_data = ActiveSceneDataSingleton()
        framebuffer = active_scene_data.framebuffer
        framebuffer.clear(red=red, green=green, blue=blue, alpha=alpha)
        self._render_with_passes(scene_config, framebuffer)

        if ConfigSingleton().write_video:
            writing_process = active_scene_data.writing_process
            assert writing_process is not None
            assert writing_process.stdin is not None
            writing_process.stdin.write(framebuffer.read(components=4))
        if ConfigSingleton().preview:
            ContextSingleton()  # ensure the singleton is generated  # TODO
            assert (window := ContextSingleton._WINDOW) is not None
            assert (window_framebuffer := ContextSingleton._WINDOW_FRAMEBUFFER) is not None
            window.clear()
            RenderProcedure.fullscreen_render_step(
                shader_str=RenderProcedure.read_shader("copy"),
                custom_macros=[],
                texture_storages=[
                    self._u_color_map_o_.write(
                        np.array(active_scene_data.color_texture)
                    )
                ],
                uniform_blocks=[],
                framebuffer=window_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.NOTHING
                )
            )
            if (previous_timestamp := self._previous_rendering_timestamp) is not None and \
                    (sleep_t := (1.0 / ConfigSingleton().fps) - (time.time() - previous_timestamp)) > 0.0:
                time.sleep(sleep_t)
            window.swap_buffers()
        self._previous_rendering_timestamp = time.time()

    def _find_frame_range(self, start_frame_floating_index: Real, stop_frame_floating_index: Real) -> range:
        # Find all frame indices in the intersection of
        # (start_frame_floating_index, stop_frame_floating_index]
        # and [ConfigSingleton().start_frame_index, ConfigSingleton().stop_frame_index]
        start_frame_index = int(np.ceil(
            start_frame_floating_index
            if (config_start_frame_index := ConfigSingleton().start_frame_index) is None
            else max(config_start_frame_index, start_frame_floating_index)
        ))
        stop_frame_index = int(np.floor(
            stop_frame_floating_index
            if (config_stop_frame_index := ConfigSingleton().stop_frame_index is None)
            else max(config_stop_frame_index, stop_frame_floating_index)
        ))
        if np.isclose(start_frame_index, start_frame_floating_index):
            # Exclude the open side
            start_frame_index += 1
        return range(start_frame_index, stop_frame_index + 1)

    def _update_dt(self, dt: Real):
        assert dt >= 0.0
        animation_dict_copy = self._animation_dict_.copy()
        for animation in list(animation_dict_copy):
            t0 = animation_dict_copy[animation]
            t = t0 + dt
            animation_dict_copy[animation] = t
            if t < animation._start_time:
                continue

            animation_expired = False
            if animation._stop_time is not None and t > animation._stop_time:
                animation_expired = True
                t = animation._stop_time

            for addition_item in animation._mobject_addition_items[:]:
                t_addition, mobject, parent = addition_item
                if t < t_addition:
                    continue
                if parent is None:
                    parent = self
                parent.add(mobject)
                animation._mobject_addition_items.remove(addition_item)

            animation._animate_func(t0, t)

            for removal_item in animation._mobject_removal_items[:]:
                t_removal, mobject, parent = removal_item
                if t < t_removal:
                    continue
                if parent is None:
                    parent = self
                parent.remove(mobject)
                animation._mobject_removal_items.remove(removal_item)

            if animation_expired:
                if animation._mobject_addition_items:
                    warnings.warn("`mobject_addition_items` is not empty after the animation finishes")
                if animation._mobject_removal_items:
                    warnings.warn("`mobject_removal_items` is not empty after the animation finishes")
                animation_dict_copy.pop(animation)

        self._animation_dict_ = LazyData(animation_dict_copy)
        return self

    def _update_frames(self, frames: Real):
        self._update_dt(frames / ConfigSingleton().fps)
        return self

    def construct(self) -> None:
        pass

    def prepare(self, *animations: Animation):
        animation_dict = self._animation_dict_.copy()
        for animation in animations:
            animation_dict[animation] = 0.0
        self._animation_dict_ = LazyData(animation_dict)
        return self

    def play(self, *animations: Animation):
        self.prepare(*animations)
        try:
            wait_time = max(t for animation in animations if (t := animation._stop_time) is not None)
        except ValueError:
            wait_time = 0.0
        self.wait(wait_time)
        return self

    def wait(self, t: Real = 1.0):
        assert t >= 0.0
        frames = t * ConfigSingleton().fps
        start_frame_floating_index = self._frame_floating_index
        stop_frame_floating_index = start_frame_floating_index + frames
        self._frame_floating_index = stop_frame_floating_index
        frame_range = self._find_frame_range(start_frame_floating_index, stop_frame_floating_index)
        if not frame_range:
            self._update_frames(frames)
            return

        self._update_frames(frame_range.start - start_frame_floating_index)
        if self._previous_rendering_timestamp is None:
            self._render_frame()

        for _ in frame_range[:-1]:
            self._update_frames(1)
            self._render_frame()
        self._update_frames(stop_frame_floating_index - (frame_range.stop - 1))
        return self

    #def add(self, *mobjects: Mobject):
    #    self._mobject_node.add(*mobjects)
    #    return self

    #def remove(self, *mobjects: Mobject):
    #    self._mobject_node.remove(*mobjects)
    #    return self

    def set_view(
        self,
        *,
        eye: Vec3T | None = None,
        target: Vec3T | None = None,
        up: Vec3T | None = None
    ):
        self._scene_config_ = LazyData(self._scene_config_.set_view(
            eye=eye,
            target=target,
            up=up
        ))
        return self

    def set_background(
        self,
        *,
        color: ColorType | None = None,
        opacity: Real | None = None
    ):
        self._scene_config_ = LazyData(self._scene_config_.set_background(
            color=color,
            opacity=opacity
        ))
        return self

    def set_ambient_light(
        self,
        *,
        color: ColorType | None = None,
        opacity: Real | None = None
    ):
        self._scene_config_ = LazyData(self._scene_config_.set_ambient_light(
            color=color,
            opacity=opacity
        ))
        return self

    def add_point_light(
        self,
        *,
        position: Vec3T | None = None,
        color: ColorType | None = None,
        opacity: Real | None = None
    ):
        self._scene_config_ = LazyData(self._scene_config_.add_point_light(
            position=position,
            color=color,
            opacity=opacity
        ))
        return self

    def set_point_light(
        self,
        *,
        index: int | None = None,
        position: Vec3T | None = None,
        color: ColorType | None = None,
        opacity: Real | None = None
    ):
        self._scene_config_ = LazyData(self._scene_config_.set_point_light(
            index=index,
            position=position,
            color=color,
            opacity=opacity
        ))
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
        self._scene_config_ = LazyData(self._scene_config_.set_style(
            background_color=background_color,
            background_opacity=background_opacity,
            ambient_light_color=ambient_light_color,
            ambient_light_opacity=ambient_light_opacity,
            point_light_position=point_light_position,
            point_light_color=point_light_color,
            point_light_opacity=point_light_opacity
        ))
        return self
