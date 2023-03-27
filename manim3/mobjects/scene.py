__all__ = ["Scene"]


import time
import warnings

import moderngl
import numpy as np
from PIL import Image

from ..animations.animation import (
    Animation,
    RegroupItem,
    RegroupVerb
)
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
from ..rendering.gl_buffer import TexturePlaceholders
from ..rendering.vertex_array import VertexArray


class EndSceneException(Exception):
    pass


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
    def _u_color_map_(cls) -> TexturePlaceholders:
        return TexturePlaceholders(
            field="sampler2D u_color_map"
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_accum_map_(cls) -> TexturePlaceholders:
        return TexturePlaceholders(
            field="sampler2D u_accum_map"
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_revealage_map_(cls) -> TexturePlaceholders:
        return TexturePlaceholders(
            field="sampler2D u_revealage_map"
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_depth_map_(cls) -> TexturePlaceholders:
        return TexturePlaceholders(
            field="sampler2D u_depth_map"
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _copy_vertex_array_(
        cls,
        _u_color_map_: TexturePlaceholders,
        _u_depth_map_: TexturePlaceholders
    ) -> VertexArray:
        return VertexArray(
            shader_filename="copy",
            custom_macros=[
                "#define COPY_DEPTH"
            ],
            texture_placeholders=[
                _u_color_map_,
                _u_depth_map_
            ]
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _oit_accum_vertex_array_(
        cls,
        _u_color_map_: TexturePlaceholders,
        _u_depth_map_: TexturePlaceholders
    ) -> VertexArray:
        return VertexArray(
            shader_filename="oit_accum",
            texture_placeholders=[
                _u_color_map_,
                _u_depth_map_
            ]
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _oit_revealage_vertex_array_(
        cls,
        _u_color_map_: TexturePlaceholders,
        _u_depth_map_: TexturePlaceholders
    ) -> VertexArray:
        return VertexArray(
            shader_filename="oit_revealage",
            texture_placeholders=[
                _u_color_map_,
                _u_depth_map_
            ]
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _oit_compose_vertex_array_(
        cls,
        _u_accum_map_: TexturePlaceholders,
        _u_revealage_map_: TexturePlaceholders
    ) -> VertexArray:
        return VertexArray(
            shader_filename="oit_compose",
            texture_placeholders=[
                _u_accum_map_,
                _u_revealage_map_
            ]
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _copy_window_vertex_array_(
        cls,
        _u_color_map_: TexturePlaceholders
    ) -> VertexArray:
        return VertexArray(
            shader_filename="copy",
            texture_placeholders=[
                _u_color_map_
            ]
        )

    def _render(
        self,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        # Inspired from `https://github.com/ambrosiogabe/MathAnimation`
        # `./Animations/src/renderer/Renderer.cpp`.
        opaque_mobjects: list[Mobject] = []
        transparent_mobjects: list[Mobject] = []
        for mobject in self.iter_descendants():
            if not mobject._has_local_sample_points_.value:
                continue
            mobject._scene_state_ = self._scene_state_
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
        render_to_video: bool = False,
        render_to_image: bool = False
    ) -> None:
        scene_state = self._scene_state_
        red, green, blue = scene_state._background_color_.value
        alpha = scene_state._background_opacity_.value
        with SimpleFramebufferBatch() as final_batch:
            framebuffer = final_batch.framebuffer
            framebuffer.clear(red=red, green=green, blue=blue, alpha=alpha)
            self._render_with_passes(framebuffer)

            if render_to_video:
                if ConfigSingleton().rendering.write_video:
                    writing_process = Context.writing_process
                    assert writing_process.stdin is not None
                    writing_process.stdin.write(final_batch.framebuffer.read(components=4))
                if ConfigSingleton().rendering.preview:
                    window = Context.window
                    if window.is_closing:
                        raise KeyboardInterrupt
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
                            (sleep_t := (1.0 / ConfigSingleton().rendering.fps) - (time.time() - previous_timestamp)) > 0.0:
                        time.sleep(sleep_t)
                    window.swap_buffers()
                self._previous_frame_rendering_timestamp = time.time()

            if render_to_image:
                image = Image.frombytes(
                    "RGBA",
                    ConfigSingleton().size.pixel_size,
                    final_batch.framebuffer.read(components=4),
                    "raw"
                ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                image.save(ConfigSingleton().path.output_dir.joinpath(f"{self.__class__.__name__}.png"))

    @classmethod
    def _find_frame_range(
        cls,
        start_frame_floating_index: float,
        stop_frame_floating_index: float
    ) -> tuple[range, bool]:
        # Find all frame indices in the intersection of
        # `(start_frame_floating_index, stop_frame_floating_index]`
        # and `[ConfigSingleton().start_frame_index, ConfigSingleton().stop_frame_index]`.
        config_start_frame_index = ConfigSingleton().rendering.start_frame_index
        config_stop_frame_index = ConfigSingleton().rendering.stop_frame_index
        start_frame_index = int(np.ceil(
            start_frame_floating_index
            if config_start_frame_index is None
            else max(config_start_frame_index, start_frame_floating_index)
        ))
        stop_frame_index = int(np.floor(
            stop_frame_floating_index
            if config_stop_frame_index is None
            else min(config_stop_frame_index, stop_frame_floating_index)
        ))
        if np.isclose(start_frame_index, start_frame_floating_index):
            # Exclude the open side.
            start_frame_index += 1
        reaches_end = config_stop_frame_index is not None and bool(np.isclose(stop_frame_index, config_stop_frame_index))
        return range(start_frame_index, stop_frame_index + 1), reaches_end

    def _regroup(
        self,
        regroup_item: RegroupItem
    ) -> None:
        mobjects = regroup_item.mobjects
        if isinstance(mobjects, Mobject | None):
            mobjects = (mobjects,)
        targets = regroup_item.targets
        if isinstance(targets, Mobject):
            targets = (targets,)
        for mobject in dict.fromkeys(mobjects):
            if mobject is None:
                mobject = self
            if regroup_item.verb == RegroupVerb.ADD:
                mobject.add(*targets)
            elif regroup_item.verb == RegroupVerb.DISCARD:
                mobject.discard(*targets)

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

            for time_regroup_item in animation._time_regroup_items[:]:
                regroup_time, regroup_item = time_regroup_item
                if t < regroup_time:
                    continue
                self._regroup(regroup_item)
                animation._time_regroup_items.remove(time_regroup_item)

            animation._animate_func(t0, t)

            if animation_expired:
                if animation._time_regroup_items:
                    warnings.warn("`time_regroup_items` is not empty after the animation finishes")
                self._animation_dict.pop(animation)

        return self

    def _update_frames(
        self,
        frames: float
    ):
        self._update_dt(frames / ConfigSingleton().rendering.fps)
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
        frames = t * ConfigSingleton().rendering.fps
        start_frame_floating_index = self._frame_floating_index
        stop_frame_floating_index = start_frame_floating_index + frames
        self._frame_floating_index = stop_frame_floating_index
        frame_range, reaches_end = self._find_frame_range(start_frame_floating_index, stop_frame_floating_index)

        if not frame_range:
            self._update_frames(frames)
        else:
            self._update_frames(frame_range.start - start_frame_floating_index)
            if self._previous_frame_rendering_timestamp is None:
                self._render_scene(render_to_video=True)
            for _ in frame_range[:-1]:
                self._update_frames(1)
                self._render_scene(render_to_video=True)
            self._update_frames(stop_frame_floating_index - (frame_range.stop - 1))

        if reaches_end:
            raise EndSceneException()
        return self

    def set_view(
        self,
        *,
        eye: Vec3T | None = None,
        target: Vec3T | None = None,
        up: Vec3T | None = None
    ):
        self._scene_state_.set_view(
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
        self._scene_state_.set_background(
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
        self._scene_state_.set_ambient_light(
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
        self._scene_state_.add_point_light(
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
        self._scene_state_.set_point_light(
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
        self._scene_state_.set_style(
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
        if ConfigSingleton().rendering.write_video:
            Context.setup_writing_process(cls.__name__)

        self = cls()

        try:
            try:
                self.construct()
            except EndSceneException:
                pass
            finally:
                if ConfigSingleton().rendering.write_last_frame:
                    self._render_scene(render_to_image=True)
        except KeyboardInterrupt:
            pass
        finally:
            if ConfigSingleton().rendering.write_video:
                writing_process = Context.writing_process
                assert writing_process.stdin is not None
                writing_process.stdin.close()
                writing_process.wait()
                writing_process.terminate()
