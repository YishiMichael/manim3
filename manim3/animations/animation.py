from abc import (
    ABC,
    abstractmethod
)
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import itertools as it
import subprocess as sp
from typing import (
    IO,
    Callable,
    Iterator
)
import weakref

import moderngl
from PIL import Image

from ..config import (
    Config,
    ConfigSingleton
)
from ..custom_typing import ColorT
from ..lazy.lazy import LazyDynamicContainer
from ..mobjects.cameras.camera import Camera
from ..mobjects.cameras.orthographic_camera import OrthographicCamera
from ..mobjects.cameras.perspective_camera import PerspectiveCamera
from ..mobjects.lighting.lighting import Lighting
from ..mobjects.mobject import Mobject
from ..mobjects.frame_mobject import FrameMobject
from ..passes.render_pass import RenderPass
from ..rendering.context import Context
from ..rendering.framebuffer import ColorFramebuffer
from ..rendering.texture import TextureFactory
from ..utils.rate import RateUtils


class TimelineState(Enum):
    START = 1
    STOP = -1
    AWAIT = 0


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TimelineSignal:
    timestamp: float
    animation: "Animation"
    absolute_rate: Callable[[float], float] | None
    timeline_state: TimelineState


class TimelineManager:
    __slots__ = (
        "start_alpha_dict",
        "signal_dict"
    )

    def __init__(self) -> None:
        super().__init__()
        self.start_alpha_dict: dict[Iterator[TimelineSignal], float] = {}
        self.signal_dict: dict[Iterator[TimelineSignal], TimelineSignal] = {}

    def add_signal_timeline(
        self,
        signal_timeline: Iterator[TimelineSignal],
        start_alpha: float
    ) -> None:
        try:
            signal = next(signal_timeline)
        except StopIteration:
            return
        self.start_alpha_dict[signal_timeline] = start_alpha
        self.signal_dict[signal_timeline] = signal

    def advance_to_next_signal(
        self,
        signal_timeline: Iterator[TimelineSignal]
    ) -> None:
        try:
            signal = next(signal_timeline)
        except StopIteration:
            self.start_alpha_dict.pop(signal_timeline)
            self.signal_dict.pop(signal_timeline)
            return
        self.signal_dict[signal_timeline] = signal

    def is_not_empty(self) -> bool:
        return bool(self.signal_dict)

    def get_next_signal_timeline_item(self) -> tuple[Iterator[TimelineSignal], float, TimelineSignal]:
        start_alpha_dict = self.start_alpha_dict
        signal_dict = self.signal_dict

        def get_next_alpha(
            signal_timeline: Iterator[TimelineSignal]
        ) -> float:
            next_alpha = start_alpha_dict[signal_timeline] + signal_dict[signal_timeline].timestamp
            return round(next_alpha, 3)  # Avoid floating error.

        signal_timeline = min(signal_dict, key=get_next_alpha)
        return signal_timeline, start_alpha_dict[signal_timeline], signal_dict[signal_timeline]


class Animation(ABC):
    __slots__ = (
        "_updater",
        "_run_time",
        "_relative_rate",
        "_delta_alpha",
        "_new_children",
        "_is_prepared_flag",
        "_scene_ref"
    )

    def __init__(
        self,
        # Update continuously (per frame).
        updater: Callable[[float], None] | None = None,
        # If provided, the animation will be clipped from right at `run_time`.
        # `parent.play(self)` will call `parent.wait(run_time)` that covers this animation.
        run_time: float | None = None,
        # `[0.0, run_time] -> [0.0, +infty), time |-> alpha`
        # Must be an increasing function.
        relative_rate: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__()
        assert run_time is None or run_time >= 0.0
        self._updater: Callable[[float], None] | None = updater
        self._run_time: float | None = run_time
        self._relative_rate: Callable[[float], float] = relative_rate
        self._delta_alpha: float = 0.0
        self._new_children: list[Animation] = []
        # The structure of animations forms a tree without any reoccurances of nodes.
        # This marks whether the node already exists in the tree.
        self._is_prepared_flag: bool = False
        self._scene_ref: weakref.ref[Scene] | None = None

    def _wait_timeline(self) -> Iterator[float]:
        timeline_coroutine = self.timeline()
        try:
            while True:
                timeline_coroutine.send(None)
                yield self._delta_alpha
        except StopIteration:
            pass

    def _signal_timeline(self) -> Iterator[TimelineSignal]:
        relative_rate = self._relative_rate
        relative_rate_inv = RateUtils.inverse(relative_rate)
        run_alpha = relative_rate(self._run_time) if self._run_time is not None else None
        current_alpha = relative_rate(0.0)
        assert current_alpha >= 0.0
        manager = TimelineManager()

        yield TimelineSignal(
            timestamp=0.0,
            animation=self,
            absolute_rate=relative_rate,
            timeline_state=TimelineState.START
        )

        for wait_delta_alpha in self._wait_timeline():
            for child in self._new_children:
                manager.add_signal_timeline(
                    signal_timeline=child._signal_timeline(),
                    start_alpha=current_alpha
                )
            self._new_children.clear()

            assert wait_delta_alpha >= 0.0
            current_alpha += wait_delta_alpha
            if run_alpha is not None and current_alpha > run_alpha:
                early_break = True
                current_alpha = run_alpha
            else:
                early_break = False

            while manager.is_not_empty():
                signal_timeline, timeline_start_alpha, signal = manager.get_next_signal_timeline_item()
                next_alpha = timeline_start_alpha + signal.timestamp
                if next_alpha > current_alpha:
                    break

                if (absolute_rate := signal.absolute_rate) is not None:
                    new_absolute_rate = RateUtils.compose(
                        RateUtils.adjust(absolute_rate, lag_time=timeline_start_alpha),
                        relative_rate
                    )
                else:
                    new_absolute_rate = None

                yield TimelineSignal(
                    timestamp=relative_rate_inv(next_alpha),
                    animation=signal.animation,
                    absolute_rate=new_absolute_rate,
                    timeline_state=signal.timeline_state
                )
                manager.advance_to_next_signal(signal_timeline)

            yield TimelineSignal(
                timestamp=relative_rate_inv(current_alpha),
                animation=self,
                absolute_rate=None,
                timeline_state=TimelineState.AWAIT
            )

            if early_break:
                break

        yield TimelineSignal(
            timestamp=relative_rate_inv(current_alpha),
            animation=self,
            absolute_rate=None,
            timeline_state=TimelineState.STOP
        )

    @property
    def _play_run_time(self) -> float:
        return run_time if (run_time := self._run_time) is not None else 0.0

    # Access the scene the animation is operated on.
    # Always accessible in the body of `timeline()` method.
    @property
    def scene(self) -> "Scene":
        assert (scene_ref := self._scene_ref) is not None
        assert (scene := scene_ref()) is not None
        return scene

    def add_to_scene(
        self,
        mobject: Mobject
    ):
        self.scene.add(mobject)
        return self

    def discard_from_scene(
        self,
        mobject: Mobject
    ):
        mobject.discarded_by(*mobject.iter_parents())
        return self

    def prepare(
        self,
        *animations: "Animation"
    ) -> None:
        for animation in animations:
            assert not animation._is_prepared_flag
            animation._is_prepared_flag = True
            if animation._scene_ref is None:
                animation._scene_ref = self._scene_ref
        self._new_children.extend(animations)

    async def wait(
        self,
        delta_alpha: float = 1.0
    ) -> None:
        self._delta_alpha = delta_alpha
        await asyncio.sleep(0.0)

    async def play(
        self,
        animation: "Animation"
    ) -> None:
        self.prepare(animation)
        await self.wait(animation._play_run_time)

    async def wait_forever(self) -> None:
        # Used for infinite animation.
        while True:
            await self.wait()

    @abstractmethod
    async def timeline(self) -> None:
        pass


class Scene(Animation):
    __slots__ = ("_scene_frame",)

    def __init__(
        self,
        start_time: float = 0.0,
        stop_time: float | None = None
    ) -> None:
        super().__init__(
            run_time=stop_time - start_time if stop_time is not None else None,
            relative_rate=RateUtils.adjust(RateUtils.linear, lag_time=-start_time)
        )
        self._scene_ref = weakref.ref(self)

        match ConfigSingleton().camera.camera_type:
            case "PerspectiveCamera":
                camera = PerspectiveCamera()
            case "OrthographicCamera":
                camera = OrthographicCamera()
        self._scene_frame: FrameMobject = FrameMobject(
            camera=camera,
            lighting=Lighting()
        )
        self.set_background(
            color=ConfigSingleton().style.background_color
        )

    async def _render(self) -> None:
        config = ConfigSingleton().rendering
        fps = config.fps
        write_video = config.write_video
        write_last_frame = config.write_last_frame
        preview = config.preview
        scene_name = type(self).__name__

        signal_timeline = self._signal_timeline()
        animation_dict: dict[Animation, Callable[[float], float]] = {}

        def animate(
            timestamp: float
        ) -> None:
            for animation, absolute_rate in animation_dict.items():
                if (updater := animation._updater) is not None:
                    updater(absolute_rate(timestamp))

        def digest_signal(
            signal: TimelineSignal
        ) -> None:
            match signal.timeline_state:
                case TimelineState.START:
                    assert signal.animation not in animation_dict
                    assert signal.absolute_rate is not None
                    animation_dict[signal.animation] = signal.absolute_rate
                case TimelineState.STOP:
                    animation_dict.pop(signal.animation)
                case TimelineState.AWAIT:
                    pass

        async def run_frame(
            clock_timestamp: float,
            signal_timestamp: float,
            framebuffer: ColorFramebuffer,
            video_stdin: IO[bytes] | None
        ) -> float | None:
            await asyncio.sleep(0.0)

            next_signal_timestamp = signal_timestamp
            while next_signal_timestamp <= clock_timestamp:
                animate(next_signal_timestamp)
                try:
                    signal = next(signal_timeline)
                except StopIteration:
                    return None
                next_signal_timestamp = signal.timestamp
                digest_signal(signal)
            animate(clock_timestamp)

            self._scene_frame._render_scene(framebuffer)
            if preview:
                self._scene_frame._render_to_window(framebuffer.color_texture)
            if video_stdin is not None:
                self._write_frame_to_video(framebuffer.color_texture, video_stdin)

            return next_signal_timestamp

        async def run_frames(
            framebuffer: ColorFramebuffer,
            video_stdin: IO[bytes] | None
        ) -> None:
            spf = 1.0 / fps
            sleep_time = spf if preview else 0.0
            signal_timestamp = 0.0
            for frame_index in it.count():
                signal_timestamp, _ = await asyncio.gather(
                    run_frame(
                        frame_index * spf,
                        signal_timestamp,
                        framebuffer,
                        video_stdin
                    ),
                    asyncio.sleep(sleep_time),
                    return_exceptions=False  #True
                )
                if signal_timestamp is None:
                    break

            self._scene_frame._render_scene(framebuffer)
            if write_last_frame:
                self._write_frame_to_image(framebuffer.color_texture, scene_name)

        Context.activate(title=scene_name, standalone=not preview)
        with TextureFactory.texture() as color_texture, \
                self._video_writer(write_video, fps, scene_name) as video_stdin:
            framebuffer = ColorFramebuffer(
                color_texture=color_texture
            )
            await run_frames(framebuffer, video_stdin)

    @classmethod
    @contextmanager
    def _video_writer(
        cls,
        write_video: bool,
        fps: float,
        scene_name: str
    ) -> Iterator[IO[bytes] | None]:
        if not write_video:
            yield None
            return
        writing_process = sp.Popen((
            "ffmpeg",
            "-y",  # Overwrite output file if it exists.
            "-f", "rawvideo",
            "-s", "{}x{}".format(*ConfigSingleton().size.pixel_size),  # size of one frame
            "-pix_fmt", "rgba",
            "-r", str(fps),  # frames per second
            "-i", "-",  # The input comes from a pipe.
            "-vf", "vflip",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            ConfigSingleton().path.output_dir.joinpath(f"{scene_name}.mp4")
        ), stdin=sp.PIPE)
        assert (video_stdin := writing_process.stdin) is not None
        yield video_stdin
        video_stdin.close()
        writing_process.wait()
        writing_process.terminate()

    @classmethod
    def _write_frame_to_video(
        cls,
        color_texture: moderngl.Texture,
        video_stdin: IO[bytes]
    ) -> None:
        video_stdin.write(color_texture.read())

    @classmethod
    def _write_frame_to_image(
        cls,
        color_texture: moderngl.Texture,
        scene_name: str
    ) -> None:
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
        return self._scene_frame._camera_

    def set_camera(
        self,
        camera: Camera
    ):
        self._scene_frame._camera_ = camera
        return self

    def render(
        self,
        config: Config | None = None
    ) -> None:
        if config is not None:
            ConfigSingleton.set(config)
        try:
            asyncio.run(self._render())
        except KeyboardInterrupt:
            pass
