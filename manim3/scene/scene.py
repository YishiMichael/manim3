import asyncio
from contextlib import contextmanager
import itertools as it
import subprocess as sp
from typing import (
    Callable,
    IO,
    Iterator
)
import weakref

import moderngl
from PIL import Image

from ..animations.animation import (
    Animation,
    TimelineSignal,
    TimelineState
)
from ..config import Config
from ..custom_typing import ColorT
from ..mobjects.mobject import Mobject
from ..mobjects.cameras.camera import Camera
from ..mobjects.scene_root_mobject import SceneRootMobject
from ..rendering.context import Context
from ..rendering.framebuffer import ColorFramebuffer
from ..rendering.texture import TextureFactory
from ..utils.rate import RateUtils


class Scene(Animation):
    __slots__ = ("_root_mobject",)

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
        self._root_mobject: SceneRootMobject = SceneRootMobject()
        self.set_background_color(
            Config().style.background_color
        )

    async def _render(self) -> None:
        config = Config().rendering
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

        def render_to_window(
            framebuffer: moderngl.Framebuffer
        ) -> None:
            window = Context.window
            if window.is_closing:
                raise KeyboardInterrupt
            window.clear()
            Context.blit(framebuffer, Context.window_framebuffer)
            window.swap_buffers()

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

            self._root_mobject._render_scene(framebuffer)
            if preview:
                render_to_window(framebuffer.framebuffer)
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

            self._root_mobject._render_scene(framebuffer)
            if write_last_frame:
                self._write_frame_to_image(framebuffer.color_texture, scene_name)

        Context.activate(title=scene_name, standalone=not preview)
        with TextureFactory.texture(components=3) as color_texture, \
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
            "-s", "{}x{}".format(*Config().size.pixel_size),  # size of one frame
            "-pix_fmt", "rgb",
            "-r", str(fps),  # frames per second
            "-i", "-",  # The input comes from a pipe.
            "-vf", "vflip",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            Config().path.output_dir.joinpath(f"{scene_name}.mp4")
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
            "RGB",
            Config().size.pixel_size,
            color_texture.read(),
            "raw"
        ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image.save(Config().path.output_dir.joinpath(f"{scene_name}.png"))

    # Shortcut access to root mobject.

    @property
    def root_mobject(self) -> SceneRootMobject:
        return self._root_mobject

    def add(
        self,
        *mobjects: "Mobject"
    ):
        self.root_mobject.add(*mobjects)
        return self

    def discard(
        self,
        *mobjects: "Mobject"
    ):
        self.root_mobject.discard(*mobjects)
        return self

    def clear(self):
        self.root_mobject.clear()
        return self

    def set_background_color(
        self,
        color: ColorT | None = None
    ):
        self.root_mobject.set_style(
            color=color,
            broadcast=False
        )
        return self

    @property
    def camera(self) -> Camera:
        return self.root_mobject._camera_

    def set_camera(
        self,
        camera: Camera
    ):
        self.root_mobject.set_style(
            camera=camera
        )
        return self

    def render(self) -> None:
        try:
            asyncio.run(self._render())
        except KeyboardInterrupt:
            pass
