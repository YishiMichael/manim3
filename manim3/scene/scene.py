import asyncio
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
        ) -> Iterator[tuple[float, float]]:
            spf = 1.0 / fps
            sleep_time = spf if sleep else 0.0
            # Do integer increment to avoid accumulated error in float addition.
            frame_index: int = 0
            while True:
                yield frame_index * spf, sleep_time
                frame_index += 1

        state_timeline = self._state_timeline()
        timeline_items: list[TimelineItem] = []

        def animate(
            timestamp: float
        ) -> None:
            for timeline_item in timeline_items:
                if (updater := timeline_item.updater) is not None:
                    updater(timeline_item.absolute_rate(timestamp))

        def digest_signal(
            signal: TimelineItemAppendSignal | TimelineItemRemoveSignal | TimelineAwaitSignal
        ) -> None:
            match signal:
                case TimelineItemAppendSignal(timeline_item=timeline_item):
                    timeline_items.append(timeline_item)
                case TimelineItemRemoveSignal(timeline_item=timeline_item):
                    timeline_items.remove(timeline_item)
                case TimelineAwaitSignal():
                    pass

        async def run_frame(
            clock_timestamp: float,
            state_timestamp: float,
            color_texture: moderngl.Texture
        ) -> float | None:
            await asyncio.sleep(0.0)

            next_state_timestamp = state_timestamp
            while next_state_timestamp <= clock_timestamp:
                animate(next_state_timestamp)
                try:
                    state = next(state_timeline)
                except StopIteration:
                    return None
                next_state_timestamp = state.timestamp
                digest_signal(state.signal)
            animate(clock_timestamp)

            self._render_to_texture(color_texture)
            if ConfigSingleton().rendering.preview:
                self._scene_frame._render_to_window(color_texture)
            if ConfigSingleton().rendering.write_video:
                self._write_to_writing_process(color_texture)

            return next_state_timestamp

        async def run_frames(
            color_texture: moderngl.Texture
        ) -> None:
            state_timestamp = 0.0
            for clock_timestamp, sleep_time in frame_clock(
                fps=ConfigSingleton().rendering.fps,
                sleep=ConfigSingleton().rendering.preview
            ):
                state_timestamp, _ = await asyncio.gather(
                    run_frame(
                        clock_timestamp,
                        state_timestamp,
                        color_texture
                    ),
                    asyncio.sleep(sleep_time)
                )
                if state_timestamp is None:
                    break

            self._render_to_texture(color_texture)
            if ConfigSingleton().rendering.write_last_frame:
                self._write_to_image(color_texture)

        with TextureFactory.texture() as color_texture:
            asyncio.run(run_frames(color_texture))

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
        *,
        color: ColorT | None = None,
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

    def set_camera(
        self,
        camera: Camera
    ):
        self._camera = camera
        return self

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
