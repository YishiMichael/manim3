from __future__ import annotations


import asyncio
import itertools
from typing import Self

from ..animatables.lights.ambient_light import AmbientLight
from ..animatables.camera import Camera
from ..animatables.lighting import Lighting
from ..constants.custom_typing import ColorType
from ..mobjects.mobject import Mobject
#from ..mobjects.scene_root_mobject import SceneRootMobject
from ..timelines.timeline.timeline import Timeline
from ..utils.color_utils import ColorUtils
from .toplevel import Toplevel


class Scene(Timeline):
    __slots__ = (
        "_camera",
        "_lighting",
        "_background_color",
        "_background_opacity",
        "_root_mobject",
        "_timestamp"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._camera: Camera = Camera()
        self._lighting: Lighting = Lighting(AmbientLight())
        self._background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._background_opacity: float = 0.0
        self._root_mobject: Mobject = Mobject()
        self._timestamp: float = 0.0
        self.set(
            background_color=Toplevel._get_config().background_color,
            background_opacity=Toplevel._get_config().background_opacity
        )

    #async def _render(
    #    self: Self
    #) -> None:

    #    async def run_frame(
    #        color_framebuffer: ColorFramebuffer,
    #        video_stdin: IO[bytes] | None
    #    ) -> None:
    #        await asyncio.sleep(0.0)
    #        if Toplevel._get_config().preview:
    #            Toplevel._get_window().pyglet_window.dispatch_events()
    #        self._progress()
    #        Toplevel._get_window().clear_event_queue()
    #        self._root_mobject._render_scene(color_framebuffer)
    #        if Toplevel._get_config().preview:
    #            self._render_to_window(color_framebuffer._framebuffer_)
    #        if video_stdin is not None:
    #            self._write_frame_to_video(color_framebuffer._color_texture_, video_stdin)

    #    async def run_frames(
    #        color_framebuffer: ColorFramebuffer,
    #        video_stdin: IO[bytes] | None
    #    ) -> None:
    #        self._root_schedule()
    #        spf = 1.0 / Toplevel._get_config().fps
    #        sleep_time = spf if Toplevel._get_config().preview else 0.0
    #        for frame_index in itertools.count():
    #            self._timestamp = frame_index * spf
    #            await asyncio.gather(
    #                run_frame(
    #                    color_framebuffer,
    #                    video_stdin
    #                ),
    #                asyncio.sleep(sleep_time),
    #                return_exceptions=False  #True
    #            )
    #            if self.get_after_terminated_state() is not None:
    #                break

    #        self._root_mobject._render_scene(color_framebuffer)
    #        if Toplevel._get_config().write_last_frame:
    #            self._write_frame_to_image(color_framebuffer._color_texture_)

    #    with self._video_writer() as video_stdin:
    #        await run_frames(ColorFramebuffer(), video_stdin)

    async def _run_frame(
        self: Self
    ) -> None:
        await asyncio.sleep(0.0)
        Toplevel._get_window()._pyglet_window.dispatch_events()
        self._progress()
        Toplevel._get_window().clear_event_queue()
        Toplevel._get_renderer()._process_frame(self)

    async def _run(
        self: Self
    ) -> None:
        self._root_schedule()
        spf = 1.0 / Toplevel._get_config().fps
        for frame_index in itertools.count():
            #sleep_time = spf if Toplevel._get_renderer()._streaming else 0.0
            self._timestamp = frame_index * spf

            async with asyncio.TaskGroup() as task_group:
                task_group.create_task(self._run_frame())
                if Toplevel._get_renderer()._streaming:
                    task_group.create_task(asyncio.sleep(spf))

            if self.get_after_terminated_state() is not None:
                break

            #await asyncio.gather(
            #    run_frame(
            #        color_framebuffer,
            #        video_stdin
            #    ),
            #    asyncio.sleep(sleep_time),
            #    return_exceptions=False  #True
            #)
            #if self.get_after_terminated_state() is not None:
            #    break

        #self._root_mobject._render_scene(color_framebuffer)
        #if Toplevel._get_config().write_last_frame:
        #    self._write_frame_to_image(color_framebuffer._color_texture_)

    #@classmethod
    #@contextmanager
    #def _video_writer(
    #    cls: type[Self]
    #) -> Iterator[IO[bytes] | None]:
    #    if not Toplevel._get_config().write_video:
    #        yield None
    #        return
    #    writing_process = subprocess.Popen((
    #        "ffmpeg",
    #        "-y",  # Overwrite output file if it exists.
    #        "-f", "rawvideo",
    #        "-s", f"{Toplevel._get_config().pixel_width}x{Toplevel._get_config().pixel_height}",  # size of one frame
    #        "-pix_fmt", "rgb24",
    #        "-r", f"{Toplevel._get_config().fps}",  # frames per second
    #        "-i", "-",  # The input comes from a pipe.
    #        "-vf", "vflip",
    #        "-an",
    #        "-vcodec", "libx264",
    #        "-pix_fmt", "yuv420p",
    #        "-loglevel", "error",
    #        PathUtils.get_output_subdir("videos").joinpath(f"{cls.__name__}.mp4")
    #    ), stdin=subprocess.PIPE)
    #    assert (video_stdin := writing_process.stdin) is not None
    #    yield video_stdin
    #    video_stdin.close()
    #    writing_process.wait()
    #    writing_process.terminate()

    #@classmethod
    #def _render_to_window(
    #    cls: type[Self],
    #    framebuffer: moderngl.Framebuffer
    #) -> None:
    #    window_framebuffer = Toplevel._get_context()._window_framebuffer
    #    Toplevel._get_context().blit(framebuffer, window_framebuffer)
    #    Toplevel._get_window().pyglet_window.flip()

    #@classmethod
    #def _write_frame_to_video(
    #    cls: type[Self],
    #    color_texture: moderngl.Texture,
    #    video_stdin: IO[bytes]
    #) -> None:
    #    video_stdin.write(color_texture.read())

    #@classmethod
    #def _write_frame_to_image(
    #    cls: type[Self],
    #    color_texture: moderngl.Texture
    #) -> None:
    #    image = Image.frombytes(
    #        "RGB",
    #        Toplevel._get_config().pixel_size,
    #        color_texture.read(),
    #        "raw"
    #    ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    #    image.save(PathUtils.get_output_subdir("images").joinpath(f"{cls.__name__}.png"))

    #@classmethod
    #def render(
    #    cls: type[Self],
    #    config: Config | None = None
    #) -> None:
    #    if config is None:
    #        config = Config()
    #    try:
    #        with Toplevel.configure(
    #            config=config,
    #            scene_cls=cls
    #        ) as self:
    #            asyncio.run(self._render())
    #    except KeyboardInterrupt:
    #        pass

    def run(
        self: Self
    ) -> None:
        with Toplevel._set_toplevel_scene(self):
            asyncio.run(self._run())

    # Shortcut access to root mobject.

    #@property
    #def root_mobject(
    #    self: Self
    #) -> SceneRootMobject:
    #    return self._root_mobject

    def add(
        self: Self,
        *mobjects: Mobject
    ) -> Self:
        self._root_mobject.add(*mobjects)
        return self

    def discard(
        self: Self,
        *mobjects: Mobject
    ) -> Self:
        self._root_mobject.discard(*mobjects)
        return self

    def clear(
        self: Self
    ) -> Self:
        self._root_mobject.clear()
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

    def set(
        self: Self,
        camera: Camera | None = None,
        lighting: Lighting | None = None,
        background_color: ColorType | None = None,
        background_opacity: float | None = None,
        *,
        broadcast: bool = True
    ) -> Self:
        if camera is not None:
            self.bind_camera(camera, broadcast=broadcast)
        if lighting is not None:
            self.bind_lighting(lighting, broadcast=broadcast)
        if background_color is not None:
            red, green, blue = tuple(float(component) for component in ColorUtils.color_to_array(background_color))
            self._background_color = (red, green, blue)
        if background_opacity is not None:
            self._background_opacity = background_opacity
        return self
