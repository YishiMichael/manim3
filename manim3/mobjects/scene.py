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
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..mobjects.mobject import Mobject
from ..rendering.config import ConfigSingleton
from ..rendering.context import ContextSingleton
from ..rendering.glsl_buffers import (
    AttributesBuffer,
    IndexBuffer,
    TextureStorage
)
from ..rendering.framebuffer_batches import (
    SceneFramebufferBatch,
    SimpleFramebufferBatch
)
from ..rendering.vertex_array import (
    ContextState,
    IndexedAttributesBuffer,
    VertexArray
)
from ..utils.active_scene_data import ActiveSceneDataSingleton
from ..utils.scene_config import SceneConfig


class Scene(Mobject):
    __slots__ = (
        "_animation_dict",
        "_frame_floating_index",
        "_previous_rendering_timestamp"
    )

    def __init__(self) -> None:
        super().__init__()
        self._animation_dict: dict[Animation, float] = {}
        # A timer scaled by fps
        self._frame_floating_index: float = 0.0
        self._previous_rendering_timestamp: float | None = None

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _color_map_(cls) -> moderngl.Texture:
        return NotImplemented

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _accum_map_(cls) -> moderngl.Texture:
        return NotImplemented

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _revealage_map_(cls) -> moderngl.Texture:
        return NotImplemented

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _depth_map_(cls) -> moderngl.Texture:
        return NotImplemented

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _indexed_attributes_buffer_(cls) -> IndexedAttributesBuffer:
        return IndexedAttributesBuffer(
            attributes=AttributesBuffer(
                fields=[
                    "vec3 in_position",
                    "vec2 in_uv"
                ],
                num_vertex=4,
                data={
                    "in_position": np.array((
                        [-1.0, -1.0, 0.0],
                        [1.0, -1.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [-1.0, 1.0, 0.0],
                    )),
                    "in_uv": np.array((
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0],
                    ))
                }
            ),
            index_buffer=IndexBuffer(
                data=np.array((
                    0, 1, 2, 3
                ))
            ),
            mode=moderngl.TRIANGLE_FAN
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_color_map_(
        cls,
        color_map: moderngl.Texture
    ) -> TextureStorage:
        return TextureStorage(
            field="sampler2D u_color_map",
            texture_array=np.array(color_map)
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_accum_map_(
        cls,
        accum_map: moderngl.Texture
    ) -> TextureStorage:
        return TextureStorage(
            field="sampler2D u_accum_map",
            texture_array=np.array(accum_map)
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_revealage_map_(
        cls,
        revealage_map: moderngl.Texture
    ) -> TextureStorage:
        return TextureStorage(
            field="sampler2D u_revealage_map",
            texture_array=np.array(revealage_map)
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_depth_map_(
        cls,
        depth_map: moderngl.Texture
    ) -> TextureStorage:
        return TextureStorage(
            field="sampler2D u_depth_map",
            texture_array=np.array(depth_map)
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _copy_vertex_array_(
        cls,
        _u_color_map_: TextureStorage,
        _u_depth_map_: TextureStorage,
        _indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_filename="copy",
            custom_macros=[
                "#define COPY_DEPTH"
            ],
            texture_storages=[
                _u_color_map_,
                _u_depth_map_
            ],
            uniform_blocks=[],
            indexed_attributes=_indexed_attributes_buffer_
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _copy_window_vertex_array_(
        cls,
        _u_color_map_: TextureStorage,
        _indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_filename="copy",
            custom_macros=[],
            texture_storages=[
                _u_color_map_
            ],
            uniform_blocks=[],
            indexed_attributes=_indexed_attributes_buffer_
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _oit_accum_vertex_array_(
        cls,
        _u_color_map_: TextureStorage,
        _u_depth_map_: TextureStorage,
        _indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_filename="oit_accum",
            custom_macros=[],
            texture_storages=[
                _u_color_map_,
                _u_depth_map_
            ],
            uniform_blocks=[],
            indexed_attributes=_indexed_attributes_buffer_
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _oit_revealage_vertex_array_(
        cls,
        _u_color_map_: TextureStorage,
        _u_depth_map_: TextureStorage,
        _indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_filename="oit_revealage",
            custom_macros=[],
            texture_storages=[
                _u_color_map_,
                _u_depth_map_
            ],
            uniform_blocks=[],
            indexed_attributes=_indexed_attributes_buffer_
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _oit_compose_vertex_array_(
        cls,
        _u_accum_map_: TextureStorage,
        _u_revealage_map_: TextureStorage,
        _indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_filename="oit_compose",
            custom_macros=[],
            texture_storages=[
                _u_accum_map_,
                _u_revealage_map_
            ],
            uniform_blocks=[],
            indexed_attributes=_indexed_attributes_buffer_
        )

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _scene_config_(cls) -> SceneConfig:
        return SceneConfig()

    def _render(
        self,
        scene_config: SceneConfig,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        # Inspired from https://github.com/ambrosiogabe/MathAnimation
        # ./Animations/src/renderer/Renderer.cpp
        opaque_mobjects: list[Mobject] = []
        transparent_mobjects: list[Mobject] = []
        for mobject in self.iter_descendants():
            if not mobject._has_local_sample_points_.value:
                continue
            if mobject._apply_oit_.value:
                transparent_mobjects.append(mobject)
            else:
                opaque_mobjects.append(mobject)

        with SceneFramebufferBatch() as scene_batch:
            for mobject in opaque_mobjects:
                with SimpleFramebufferBatch() as batch:
                    mobject._render_with_passes(scene_config, batch.framebuffer)
                    self._color_map_ = batch.color_texture
                    self._depth_map_ = batch.depth_texture
                    self._copy_vertex_array_.render(
                        #shader_filename="copy",
                        #custom_macros=[
                        #    "#define COPY_DEPTH"
                        #],
                        #texture_storages=[
                        #    self._u_color_map_,
                        #    self._u_depth_map_
                        #],
                        #uniform_blocks=[],
                        framebuffer=scene_batch.opaque_framebuffer,
                        context_state=ContextState(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=(moderngl.ONE, moderngl.ZERO)
                        )
                    )

            for mobject in transparent_mobjects:
                with SimpleFramebufferBatch() as batch:
                    mobject._render_with_passes(scene_config, batch.framebuffer)
                    self._color_map_ = batch.color_texture
                    self._depth_map_ = batch.depth_texture
                    self._oit_accum_vertex_array_.render(
                        #shader_filename="oit_accum",
                        #custom_macros=[],
                        #texture_storages=[
                        #    self._u_color_map_,
                        #    self._u_depth_map_
                        #],
                        #uniform_blocks=[],
                        framebuffer=scene_batch.accum_framebuffer,
                        context_state=ContextState(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=moderngl.ADDITIVE_BLENDING
                        )
                    )
                    self._oit_revealage_vertex_array_.render(
                        #shader_filename="oit_revealage",
                        #custom_macros=[],
                        #texture_storages=[
                        #    self._u_color_map_,
                        #    self._u_depth_map_
                        #],
                        #uniform_blocks=[],
                        framebuffer=scene_batch.revealage_framebuffer,
                        context_state=ContextState(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=(moderngl.ZERO, moderngl.ONE_MINUS_SRC_COLOR)
                        )
                    )

            self._color_map_ = scene_batch.opaque_texture
            self._depth_map_ = scene_batch.depth_texture
            self._copy_vertex_array_.render(
                #shader_filename="copy",
                #custom_macros=[
                #    "#define COPY_DEPTH"
                #],
                #texture_storages=[
                #    self._u_color_map_,
                #    self._u_depth_map_
                #],
                #uniform_blocks=[],
                framebuffer=target_framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                    blend_func=(moderngl.ONE, moderngl.ZERO)
                )
            )
            self._accum_map_ = scene_batch.accum_texture
            self._revealage_map_ = scene_batch.revealage_texture
            self._oit_compose_vertex_array_.render(
                #shader_filename="oit_compose",
                #custom_macros=[],
                #texture_storages=[
                #    self._u_accum_map_,
                #    self._u_revealage_map_
                #],
                #uniform_blocks=[],
                framebuffer=target_framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.BLEND | moderngl.DEPTH_TEST
                )
            )

    def _render_frame(self) -> None:
        scene_config = self._scene_config_
        red, green, blue = scene_config._background_color_.value
        alpha = scene_config._background_opacity_.value

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
            self._color_map_ = active_scene_data.color_texture
            self._copy_window_vertex_array_.render(
                #shader_filename="copy",
                #custom_macros=[],
                #texture_storages=[
                #    self._u_color_map_
                #],
                #uniform_blocks=[],
                framebuffer=window_framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.NOTHING
                )
            )
            if (previous_timestamp := self._previous_rendering_timestamp) is not None and \
                    (sleep_t := (1.0 / ConfigSingleton().fps) - (time.time() - previous_timestamp)) > 0.0:
                time.sleep(sleep_t)
            window.swap_buffers()
        self._previous_rendering_timestamp = time.time()

    @classmethod
    def _find_frame_range(
        cls,
        start_frame_floating_index: Real,
        stop_frame_floating_index: Real
    ) -> range:
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

    def _update_dt(
        self,
        dt: Real
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
        frames: Real
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
        t: Real = 1.0
    ):
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
        opacity: Real | None = None
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
        opacity: Real | None = None
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
        opacity: Real | None = None
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
        opacity: Real | None = None
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
        background_opacity: Real | None = None,
        ambient_light_color: ColorType | None = None,
        ambient_light_opacity: Real | None = None,
        point_light_position: Vec3T | None = None,
        point_light_color: ColorType | None = None,
        point_light_opacity: Real | None = None
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
