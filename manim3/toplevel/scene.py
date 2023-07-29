import asyncio
import itertools as it
import subprocess as sp
from contextlib import contextmanager
from typing import (
    IO,
    Iterator
)

import moderngl
from PIL import Image

from ..animations.animation.animation import Animation
from ..animations.animation.animation_state import AnimationState
from ..mobjects.cameras.camera import Camera
from ..mobjects.cameras.perspective_camera import PerspectiveCamera
from ..mobjects.mobject.mobject import Mobject
from ..mobjects.lights.ambient_light import AmbientLight
from ..mobjects.lights.lighting import Lighting
from ..mobjects.scene_root_mobject import SceneRootMobject
from ..rendering.framebuffers.color_framebuffer import ColorFramebuffer
from ..utils.color import ColorUtils
from ..utils.path import PathUtils
from .config import Config
from .toplevel import Toplevel


class Scene(Animation):
    __slots__ = (
        "_camera",
        "_lighting",
        "_root_mobject",
        "_timestamp"
    )

    def __init__(self) -> None:
        super().__init__()
        self._camera: Camera = PerspectiveCamera()
        self._lighting: Lighting = Lighting(AmbientLight())
        self._root_mobject: SceneRootMobject = SceneRootMobject(
            background_color=ColorUtils.standardize_color(Toplevel.config.background_color)
        )
        self._timestamp: float = 0.0

    async def _render(self) -> None:
        config = Toplevel.config
        fps = config.fps
        write_video = config.write_video
        write_last_frame = config.write_last_frame
        preview = config.preview

        async def run_frame(
            color_framebuffer: ColorFramebuffer,
            video_stdin: IO[bytes] | None
        ) -> None:
            await asyncio.sleep(0.0)
            if preview:
                Toplevel.window.dispatch_events()
            self._progress()
            Toplevel.event_queue.clear()
            self._root_mobject._render_scene(color_framebuffer)
            if preview:
                self._render_to_window(color_framebuffer._framebuffer_)
            if video_stdin is not None:
                self._write_frame_to_video(color_framebuffer._color_texture_, video_stdin)

        async def run_frames(
            color_framebuffer: ColorFramebuffer,
            video_stdin: IO[bytes] | None
        ) -> None:
            self._schedule()
            spf = 1.0 / fps
            sleep_time = spf if preview else 0.0
            for frame_index in it.count():
                self._timestamp = frame_index * spf
                await asyncio.gather(
                    run_frame(
                        color_framebuffer,
                        video_stdin
                    ),
                    asyncio.sleep(sleep_time),
                    return_exceptions=False  #True
                )
                if self._animation_state == AnimationState.AFTER_ANIMATION:
                    break

            self._root_mobject._render_scene(color_framebuffer)
            if write_last_frame:
                self._write_frame_to_image(color_framebuffer._color_texture_)

        with self._video_writer(write_video, fps) as video_stdin:
            await run_frames(ColorFramebuffer(), video_stdin)

    @classmethod
    @contextmanager
    def _video_writer(
        cls,
        write_video: bool,
        fps: int
    ) -> Iterator[IO[bytes] | None]:
        if not write_video:
            yield None
            return
        writing_process = sp.Popen((
            "ffmpeg",
            "-y",  # Overwrite output file if it exists.
            "-f", "rawvideo",
            "-s", "{}x{}".format(*Toplevel.config.pixel_size),  # size of one frame
            "-pix_fmt", "rgb24",
            "-r", str(fps),  # frames per second
            "-i", "-",  # The input comes from a pipe.
            "-vf", "vflip",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            PathUtils.output_dir.joinpath(f"{cls.__name__}.mp4")
        ), stdin=sp.PIPE)
        assert (video_stdin := writing_process.stdin) is not None
        yield video_stdin
        video_stdin.close()
        writing_process.wait()
        writing_process.terminate()

    @classmethod
    def _render_to_window(
        cls,
        framebuffer: moderngl.Framebuffer
    ) -> None:
        window_framebuffer = Toplevel.context._window_framebuffer
        Toplevel.context.blit(framebuffer, window_framebuffer)
        Toplevel.window.flip()

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
        color_texture: moderngl.Texture
    ) -> None:
        image = Image.frombytes(
            "RGB",
            Toplevel.config.pixel_size,
            color_texture.read(),
            "raw"
        ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image.save(PathUtils.output_dir.joinpath(f"{cls.__name__}.png"))

    @classmethod
    def render(
        cls,
        config: Config | None = None
    ) -> None:
        #self = cls()
        if config is None:
            config = Config()
        try:
            with Toplevel.configure(
                config=config,
                scene_cls=cls
            ) as self:
                asyncio.run(self._render())
        except KeyboardInterrupt:
            pass

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

    @property
    def camera(self) -> Camera:
        return self._camera

    @property
    def lighting(self) -> Lighting:
        return self._lighting
