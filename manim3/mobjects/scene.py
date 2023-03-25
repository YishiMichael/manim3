__all__ = ["Scene"]


import time
from typing import Callable
import warnings

import moderngl
import numpy as np
from PIL import Image

from ..animations.animation import Animation
from ..custom_typing import (
    ColorType,
    Vec3T
)
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..mobjects.mobject import Mobject
from ..rendering.config import (
    Config,
    ConfigSingleton
)
from ..rendering.context import (
    Context,
    ContextState
)
from ..rendering.framebuffer_batch import (
    SceneFramebufferBatch,
    SimpleFramebufferBatch
)
from ..rendering.gl_buffer import TextureStorage
from ..rendering.vertex_array import VertexArray


class Scene(Mobject):
    __slots__ = (
        "_animation_dict",
        "_frame_floating_index",
        "_previous_frame_rendering_timestamp"
    )

    def __init__(self) -> None:
        super().__init__()
        self._animation_dict: dict[Animation, float] = {}
        # A timer scaled by fps.
        self._frame_floating_index: float = 0.0
        self._previous_frame_rendering_timestamp: float | None = None

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_color_map_(cls) -> TextureStorage:
        return TextureStorage(
            field="sampler2D u_color_map"
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_accum_map_(cls) -> TextureStorage:
        return TextureStorage(
            field="sampler2D u_accum_map"
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_revealage_map_(cls) -> TextureStorage:
        return TextureStorage(
            field="sampler2D u_revealage_map"
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_depth_map_(cls) -> TextureStorage:
        return TextureStorage(
            field="sampler2D u_depth_map"
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _copy_vertex_array_(
        cls,
        _u_color_map_: TextureStorage,
        _u_depth_map_: TextureStorage
    ) -> VertexArray:
        return VertexArray(
            shader_filename="copy",
            custom_macros=[
                "#define COPY_DEPTH"
            ],
            texture_storages=[
                _u_color_map_,
                _u_depth_map_
            ]
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _oit_accum_vertex_array_(
        cls,
        _u_color_map_: TextureStorage,
        _u_depth_map_: TextureStorage
    ) -> VertexArray:
        return VertexArray(
            shader_filename="oit_accum",
            texture_storages=[
                _u_color_map_,
                _u_depth_map_
            ]
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _oit_revealage_vertex_array_(
        cls,
        _u_color_map_: TextureStorage,
        _u_depth_map_: TextureStorage
    ) -> VertexArray:
        return VertexArray(
            shader_filename="oit_revealage",
            texture_storages=[
                _u_color_map_,
                _u_depth_map_
            ]
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _oit_compose_vertex_array_(
        cls,
        _u_accum_map_: TextureStorage,
        _u_revealage_map_: TextureStorage
    ) -> VertexArray:
        return VertexArray(
            shader_filename="oit_compose",
            texture_storages=[
                _u_accum_map_,
                _u_revealage_map_
            ]
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _copy_window_vertex_array_(
        cls,
        _u_color_map_: TextureStorage
    ) -> VertexArray:
        return VertexArray(
            shader_filename="copy",
            texture_storages=[
                _u_color_map_
            ]
        )

    def _render(
        self,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        # Inspired from https://github.com/ambrosiogabe/MathAnimation
        # ./Animations/src/renderer/Renderer.cpp
        opaque_mobjects: list[Mobject] = []
        transparent_mobjects: list[Mobject] = []
        for mobject in self.iter_descendants():
            if not mobject._has_local_sample_points_.value:
                continue
            mobject._scene_config_ = self._scene_config_
            if mobject._apply_oit_.value:
                transparent_mobjects.append(mobject)
            else:
                opaque_mobjects.append(mobject)

        with SceneFramebufferBatch() as scene_batch:
            for mobject in opaque_mobjects:
                with SimpleFramebufferBatch() as batch:
                    mobject._render_with_passes(batch.framebuffer)
                    self._copy_vertex_array_.render(
                        texture_array_dict={
                            "u_color_map": np.array(batch.color_texture),
                            "u_depth_map": np.array(batch.depth_texture)
                        },
                        framebuffer=scene_batch.opaque_framebuffer,
                        context_state=ContextState(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=(moderngl.ONE, moderngl.ZERO)
                        )
                    )

            for mobject in transparent_mobjects:
                with SimpleFramebufferBatch() as batch:
                    mobject._render_with_passes(batch.framebuffer)
                    self._oit_accum_vertex_array_.render(
                        texture_array_dict={
                            "u_color_map": np.array(batch.color_texture),
                            "u_depth_map": np.array(batch.depth_texture)
                        },
                        framebuffer=scene_batch.accum_framebuffer,
                        context_state=ContextState(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=moderngl.ADDITIVE_BLENDING
                        )
                    )
                    self._oit_revealage_vertex_array_.render(
                        texture_array_dict={
                            "u_color_map": np.array(batch.color_texture),
                            "u_depth_map": np.array(batch.depth_texture)
                        },
                        framebuffer=scene_batch.revealage_framebuffer,
                        context_state=ContextState(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=(moderngl.ZERO, moderngl.ONE_MINUS_SRC_COLOR)
                        )
                    )

            self._copy_vertex_array_.render(
                texture_array_dict={
                    "u_color_map": np.array(scene_batch.opaque_texture),
                    "u_depth_map": np.array(scene_batch.depth_texture)
                },
                framebuffer=target_framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                    blend_func=(moderngl.ONE, moderngl.ZERO)
                )
            )
            self._oit_compose_vertex_array_.render(
                texture_array_dict={
                    "u_accum_map": np.array(scene_batch.accum_texture),
                    "u_revealage_map": np.array(scene_batch.revealage_texture)
                },
                framebuffer=target_framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.BLEND | moderngl.DEPTH_TEST
                )
            )

    def _render_scene(
        self,
        final_batch_callback: "Callable[[Scene, SimpleFramebufferBatch], None]"
    ) -> None:
        scene_config = self._scene_config_
        red, green, blue = scene_config._background_color_.value
        alpha = scene_config._background_opacity_.value
        with SimpleFramebufferBatch() as final_batch:
            framebuffer = final_batch.framebuffer
            framebuffer.clear(red=red, green=green, blue=blue, alpha=alpha)
            self._render_with_passes(framebuffer)
            final_batch_callback(self, final_batch)

    def _render_to_video_frame_callback(
        self,
        final_batch: SimpleFramebufferBatch
    ) -> None:
        if ConfigSingleton().write_video:
            writing_process = Context.writing_process
            assert writing_process.stdin is not None
            writing_process.stdin.write(final_batch.framebuffer.read(components=4))
        if ConfigSingleton().preview:
            window = Context.window
            window.clear()
            self._copy_window_vertex_array_.render(
                texture_array_dict={
                    "u_color_map": np.array(final_batch.color_texture)
                },
                framebuffer=Context.window_framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.NOTHING
                )
            )
            if (previous_timestamp := self._previous_frame_rendering_timestamp) is not None and \
                    (sleep_t := (1.0 / ConfigSingleton().fps) - (time.time() - previous_timestamp)) > 0.0:
                time.sleep(sleep_t)
            window.swap_buffers()
        self._previous_frame_rendering_timestamp = time.time()

    def _render_to_video_frame(self) -> None:
        self._render_scene(self.__class__._render_to_video_frame_callback)

    def _render_to_image_callback(
        self,
        final_batch: SimpleFramebufferBatch
    ) -> None:
        # TODO: the image is flipped in y direction
        image = Image.frombytes(
            "RGBA",
            ConfigSingleton().pixel_size,
            final_batch.framebuffer.read(components=4),
            "raw"
        )
        image.save(ConfigSingleton().output_dir.joinpath(f"{self.__class__.__name__}.png"))

    def _render_to_image(self) -> None:
        self._render_scene(self.__class__._render_to_image_callback)

    @classmethod
    def _find_frame_range(
        cls,
        start_frame_floating_index: float,
        stop_frame_floating_index: float
    ) -> range:
        # Find all frame indices in the intersection of
        # `(start_frame_floating_index, stop_frame_floating_index]`
        # and `[ConfigSingleton().start_frame_index, ConfigSingleton().stop_frame_index]`.
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
            # Exclude the open side.
            start_frame_index += 1
        return range(start_frame_index, stop_frame_index + 1)

    def _update_dt(
        self,
        dt: float
    ):
        assert dt >= 0.0
        for animation in list(self._animation_dict):
            t0 = self._animation_dict[animation]
            t = t0 + dt
            self._animation_dict[animation] = t
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
                self._animation_dict.pop(animation)

        return self

    def _update_frames(
        self,
        frames: float
    ):
        self._update_dt(frames / ConfigSingleton().fps)
        return self

    def construct(self) -> None:
        pass

    def prepare(
        self,
        *animations: Animation
    ):
        for animation in animations:
            self._animation_dict[animation] = 0.0
        return self

    def play(
        self,
        *animations: Animation
    ):
        self.prepare(*animations)
        try:
            wait_time = max(t for animation in animations if (t := animation._stop_time) is not None)
        except ValueError:
            wait_time = 0.0
        self.wait(wait_time)
        return self

    def wait(
        self,
        t: float = 1.0
    ):
        assert t >= 0.0
        frames = t * ConfigSingleton().fps
        start_frame_floating_index = self._frame_floating_index
        stop_frame_floating_index = start_frame_floating_index + frames
        self._frame_floating_index = stop_frame_floating_index
        frame_range = self._find_frame_range(start_frame_floating_index, stop_frame_floating_index)
        if not frame_range:
            self._update_frames(frames)
            return self

        self._update_frames(frame_range.start - start_frame_floating_index)
        if self._previous_frame_rendering_timestamp is None:
            self._render_to_video_frame()

        for _ in frame_range[:-1]:
            self._update_frames(1)
            self._render_to_video_frame()
        self._update_frames(stop_frame_floating_index - (frame_range.stop - 1))
        return self

    def set_view(
        self,
        *,
        eye: Vec3T | None = None,
        target: Vec3T | None = None,
        up: Vec3T | None = None
    ):
        self._scene_config_.set_view(
            eye=eye,
            target=target,
            up=up
        )
        return self

    def set_background(
        self,
        *,
        color: ColorType | None = None,
        opacity: float | None = None
    ):
        self._scene_config_.set_background(
            color=color,
            opacity=opacity
        )
        return self

    def set_ambient_light(
        self,
        *,
        color: ColorType | None = None,
        opacity: float | None = None
    ):
        self._scene_config_.set_ambient_light(
            color=color,
            opacity=opacity
        )
        return self

    def add_point_light(
        self,
        *,
        position: Vec3T | None = None,
        color: ColorType | None = None,
        opacity: float | None = None
    ):
        self._scene_config_.add_point_light(
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
        opacity: float | None = None
    ):
        self._scene_config_.set_point_light(
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
        background_opacity: float | None = None,
        ambient_light_color: ColorType | None = None,
        ambient_light_opacity: float | None = None,
        point_light_position: Vec3T | None = None,
        point_light_color: ColorType | None = None,
        point_light_opacity: float | None = None
    ):
        self._scene_config_.set_style(
            background_color=background_color,
            background_opacity=background_opacity,
            ambient_light_color=ambient_light_color,
            ambient_light_opacity=ambient_light_opacity,
            point_light_position=point_light_position,
            point_light_color=point_light_color,
            point_light_opacity=point_light_opacity
        )
        return self

    @classmethod
    def render(
        cls,
        config: Config | None = None
    ) -> None:
        if config is None:
            config = Config()

        ConfigSingleton.set(config)
        Context.activate()
        if ConfigSingleton().write_video:
            Context.setup_writing_process(cls.__name__)

        self = cls()
        self.construct()

        if ConfigSingleton().write_video:
            writing_process = Context.writing_process
            assert writing_process.stdin is not None
            writing_process.stdin.close()
            writing_process.wait()
            writing_process.terminate()
        if ConfigSingleton().write_last_frame:
            self._render_to_image()
