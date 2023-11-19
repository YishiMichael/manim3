from __future__ import annotations


import asyncio
import itertools
import subprocess
from contextlib import contextmanager
from typing import (
    IO,
    Iterator,
    Self
)

import moderngl
from PIL import Image

from ..animatables.lights.ambient_light import AmbientLight
from ..animatables.camera import Camera
from ..animatables.lighting import Lighting
from ..timelines.timeline.timeline import Timeline
from ..mobjects.scene_root_mobject import SceneRootMobject
from ..mobjects.mobject import Mobject
from ..rendering.framebuffers.color_framebuffer import ColorFramebuffer
from ..utils.path_utils import PathUtils
from .config import Config
from .toplevel import Toplevel


class Scene(Timeline):
    __slots__ = (
        "_camera",
        "_lighting",
        "_root_mobject",
        "_timestamp"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._camera: Camera = Camera()
        self._lighting: Lighting = Lighting(AmbientLight())
        self._root_mobject: SceneRootMobject = SceneRootMobject()
        self._timestamp: float = 0.0

    async def _render(
        self: Self
    ) -> None:
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
                Toplevel.window.pyglet_window.dispatch_events()
            self._progress()
            Toplevel.window.event_queue.clear()
            self._root_mobject._render_scene(color_framebuffer)
            if preview:
                self._render_to_window(color_framebuffer._framebuffer_)
            if video_stdin is not None:
                self._write_frame_to_video(color_framebuffer._color_texture_, video_stdin)

        async def run_frames(
            color_framebuffer: ColorFramebuffer,
            video_stdin: IO[bytes] | None
        ) -> None:
            self._root_schedule()
            spf = 1.0 / fps
            sleep_time = spf if preview else 0.0
            for frame_index in itertools.count():
                self._timestamp = frame_index * spf
                await asyncio.gather(
                    run_frame(
                        color_framebuffer,
                        video_stdin
                    ),
                    asyncio.sleep(sleep_time),
                    return_exceptions=False  #True
                )
                if self.get_after_terminated_state() is not None:
                    break

            self._root_mobject._render_scene(color_framebuffer)
            if write_last_frame:
                self._write_frame_to_image(color_framebuffer._color_texture_)

        with self._video_writer(write_video, fps) as video_stdin:
            await run_frames(ColorFramebuffer(), video_stdin)

    @classmethod
    @contextmanager
    def _video_writer(
        cls: type[Self],
        write_video: bool,
        fps: int
    ) -> Iterator[IO[bytes] | None]:
        if not write_video:
            yield None
            return
        writing_process = subprocess.Popen((
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
            PathUtils.get_output_subdir("videos").joinpath(f"{cls.__name__}.mp4")
        ), stdin=subprocess.PIPE)
        assert (video_stdin := writing_process.stdin) is not None
        yield video_stdin
        video_stdin.close()
        writing_process.wait()
        writing_process.terminate()

    @classmethod
    def _render_to_window(
        cls: type[Self],
        framebuffer: moderngl.Framebuffer
    ) -> None:
        window_framebuffer = Toplevel.context._window_framebuffer
        Toplevel.context.blit(framebuffer, window_framebuffer)
        Toplevel.window.pyglet_window.flip()

    @classmethod
    def _write_frame_to_video(
        cls: type[Self],
        color_texture: moderngl.Texture,
        video_stdin: IO[bytes]
    ) -> None:
        video_stdin.write(color_texture.read())

    @classmethod
    def _write_frame_to_image(
        cls: type[Self],
        color_texture: moderngl.Texture
    ) -> None:
        image = Image.frombytes(
            "RGB",
            Toplevel.config.pixel_size,
            color_texture.read(),
            "raw"
        ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image.save(PathUtils.get_output_subdir("images").joinpath(f"{cls.__name__}.png"))

    @classmethod
    def render(
        cls: type[Self],
        config: Config | None = None
    ) -> None:
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
    def root_mobject(
        self: Self
    ) -> SceneRootMobject:
        return self._root_mobject

    def add(
        self: Self,
        *mobjects: Mobject
    ) -> Self:
        self.root_mobject.add(*mobjects)
        return self

    def discard(
        self: Self,
        *mobjects: Mobject
    ) -> Self:
        self.root_mobject.discard(*mobjects)
        return self

    def clear(
        self: Self
    ) -> Self:
        self.root_mobject.clear()
        return self

    @property
    def camera(
        self: Self
    ) -> Camera:
        return self._camera

    @property
    def lighting(
        self: Self
    ) -> Lighting:
        return self._lighting

    def bind_camera(
        self: Self,
        camera: Camera,
        *,
        broadcast: bool = True
    ) -> Self:
        self._camera = camera
        self._root_mobject.bind_camera(camera, broadcast=broadcast)
        return self

    def bind_lighting(
        self: Self,
        lighting: Lighting,
        *,
        broadcast: bool = True
    ) -> Self:
        self._lighting = lighting
        self._root_mobject.bind_lighting(lighting, broadcast=broadcast)
        return self
