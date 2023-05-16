import time
from typing import Iterator

import moderngl
from PIL import Image

from ..cameras.camera import Camera
from ..cameras.perspective_camera import PerspectiveCamera
from ..config import (
    Config,
    ConfigSingleton
)

from ..custom_typing import ColorT
from ..lazy.lazy import LazyDynamicContainer
from ..lighting.ambient_light import AmbientLight
from ..lighting.lighting import Lighting
from ..lighting.point_light import PointLight
from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.mobject import Mobject
from ..mobjects.scene_frame import SceneFrame
from ..passes.render_pass import RenderPass
from ..rendering.context import Context
from ..rendering.framebuffer import ColorFramebuffer
from ..rendering.texture import TextureFactory
from ..scene.timeline import (
    Timeline,
    TimelineAwaitSignal,
    TimelineItem,
    TimelineItemAppendSignal,
    TimelineItemRemoveSignal
)
from ..utils.rate import RateUtils


class Scene(Timeline):
    __slots__ = (
        "_scene_frame",
        "_camera",
        "_lighting"
    )

    def __init__(self) -> None:
        start_time = ConfigSingleton().rendering.start_time
        stop_time = ConfigSingleton().rendering.stop_time
        super().__init__(
            run_time=stop_time - start_time if stop_time is not None else None,
            relative_rate=RateUtils.adjust(RateUtils.linear, lag_time=-start_time)
        )
        self._scene_frame: SceneFrame = SceneFrame()
        self._camera: Camera = PerspectiveCamera()
        self._lighting: Lighting = Lighting()
        self._scene_frame.set_style(
            color=ConfigSingleton().rendering.background_color
        )

    def _run_timeline(self) -> None:

        def frame_clock(
            fps: float,
            sleep: bool
        ) -> Iterator[float]:
            spf = 1.0 / fps
            # Do integer increment to avoid accumulated error in float addition.
            frame_index: int = 0
            prev_real_time: float = 0.0
            while True:
                yield frame_index * spf
                frame_index += 1
                real_time = time.time()
                if sleep and (sleep_time := spf - (real_time - prev_real_time)) > 0.0:
                    time.sleep(sleep_time)
                prev_real_time = time.time()

        def animate(
            timestamp: float,
            timeline_items: list[TimelineItem]
        ) -> None:
            for timeline_item in timeline_items:
                if (updater := timeline_item.timeline._updater) is not None:
                    updater(timeline_item.absolute_rate(timestamp))

        #AnimationItems.activate()
        absolute_timeline = self._absolute_timeline()
        state_final_timestamp = 0.0
        timeline_items: list[TimelineItem] = []
        terminated: bool = False
        with TextureFactory.texture() as color_texture:
            for timestamp in frame_clock(
                fps=ConfigSingleton().rendering.fps,
                sleep=ConfigSingleton().rendering.preview
            ):
                while state_final_timestamp <= timestamp:
                    #timeline_items.animate(state_final_timestamp)
                    animate(state_final_timestamp, timeline_items)
                    try:
                        state = next(absolute_timeline)
                    except StopIteration:
                        terminated = True
                        break
                    state_final_timestamp = state.timestamp
                    #timeline_items.digest_signal(state.signal)
                    signal = state.signal
                    if isinstance(signal, TimelineItemAppendSignal):
                        timeline_items.append(signal.timeline_item)
                    elif isinstance(signal, TimelineItemRemoveSignal):
                        timeline_items.remove(signal.timeline_item)
                    elif isinstance(signal, TimelineAwaitSignal):
                        pass
                    else:
                        raise TypeError
                if terminated:
                    break
                #timeline_items.animate(timestamp)
                animate(timestamp, timeline_items)

                self._render_to_texture(color_texture)
                if ConfigSingleton().rendering.preview:
                    self._scene_frame._render_to_window(color_texture)
                if ConfigSingleton().rendering.write_video:
                    self._write_to_writing_process(color_texture)

            self._render_to_texture(color_texture)
            if ConfigSingleton().rendering.write_last_frame:
                self._write_to_image(color_texture)

    def _render_to_texture(
        self,
        color_texture: moderngl.Texture
    ) -> None:
        scene_frame = self._scene_frame

        camera = self._camera
        for mobject in scene_frame.iter_descendants_by_type(mobject_type=Mobject):
            mobject._camera_ = camera

        lighting = self._lighting
        lighting._ambient_lights_ = scene_frame.iter_descendants_by_type(mobject_type=AmbientLight)
        lighting._point_lights_ = scene_frame.iter_descendants_by_type(mobject_type=PointLight)
        for mobject in scene_frame.iter_descendants_by_type(mobject_type=MeshMobject):
            mobject._lighting_ = lighting

        framebuffer = ColorFramebuffer(
            color_texture=color_texture
        )
        scene_frame._render_scene_with_passes(framebuffer)

    @classmethod
    def _write_to_writing_process(
        cls,
        color_texture: moderngl.Texture
    ) -> None:
        writing_process = Context.writing_process
        assert writing_process.stdin is not None
        writing_process.stdin.write(color_texture.read())

    @classmethod
    def _write_to_image(
        cls,
        color_texture: moderngl.Texture
    ) -> None:
        scene_name = ConfigSingleton().rendering.scene_name
        image = Image.frombytes(
            "RGBA",
            ConfigSingleton().size.pixel_size,
            color_texture.read(),
            "raw"
        ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image.save(ConfigSingleton().path.output_dir.joinpath(f"{scene_name}.png"))

    def add(
        self,
        *mobjects: "Mobject"
    ):
        self._scene_frame.add(*mobjects)
        return self

    def discard(
        self,
        *mobjects: "Mobject"
    ):
        self._scene_frame.discard(*mobjects)
        return self

    def clear(self):
        self._scene_frame.clear()
        return self

    def set_background(
        self,
        color: ColorT | None = None,
        *,
        opacity: float | None = None
    ):
        self._scene_frame.set_style(
            color=color,
            opacity=opacity,
            broadcast=False
        )
        return self

    @property
    def render_passes(self) -> LazyDynamicContainer[RenderPass]:
        return self._scene_frame._render_passes_

    @property
    def camera(self) -> Camera:
        return self._camera

    @classmethod
    def render(
        cls,
        config: Config | None = None
    ) -> None:
        if config is None:
            config = Config()

        ConfigSingleton.set(config)
        if ConfigSingleton().rendering.scene_name is NotImplemented:
            ConfigSingleton().rendering.scene_name = cls.__name__

        Context.activate()
        if ConfigSingleton().rendering.write_video:
            Context.setup_writing_process()

        self = cls()

        try:
            self._run_timeline()
        except KeyboardInterrupt:
            pass
        finally:
            if ConfigSingleton().rendering.write_video:
                Context.terminate_writing_process()
