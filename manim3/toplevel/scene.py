import asyncio
from contextlib import contextmanager
import itertools as it
import subprocess as sp
from typing import (
    Callable,
    IO,
    Iterator
)
#import weakref

import moderngl
from PIL import Image

#from ..animations.animation import (
#    Animation,
#    TimelineSignal,
#    TimelineStartSignal,
#    TimelineStopSignal,
#    Toplevel  # TODO
#)
from ..animations.animation import Animation
from ..constants.custom_typing import ColorT
from ..mobjects.cameras.camera import Camera
from ..mobjects.cameras.perspective_camera import PerspectiveCamera
from ..mobjects.mobject import Mobject
from ..mobjects.lights.ambient_light import AmbientLight
from ..mobjects.lights.lighting import Lighting
from ..mobjects.scene_root_mobject import SceneRootMobject
from ..rendering.framebuffers.color_framebuffer import ColorFramebuffer
#from ..scene.config import Config
from ..utils.color import ColorUtils
from ..utils.path import PathUtils
from ..utils.rate import RateUtils
from .config import Config
from .toplevel import Toplevel


class Scene(Animation):
    __slots__ = (
        "_camera",
        "_lighting",
        "_root_mobject"
        #"_framebuffer"
    )

    def __init__(
        self,
        start_time: float = 0.0,
        stop_time: float | None = None,
        *,
        camera: Camera | None = None,
        lighting: Lighting | None = None,
        background_color: ColorT | None = None
    ) -> None:
        super().__init__(
            run_time=stop_time - start_time if stop_time is not None else None,
            relative_rate=RateUtils.adjust(RateUtils.linear, lag_alpha=-start_time)
        )

        if camera is None:
            camera = PerspectiveCamera()
        if lighting is None:
            lighting = Lighting(AmbientLight())
        if background_color is None:
            background_color = Toplevel.config.background_color

        self._camera: Camera = camera
        self._lighting: Lighting = lighting
        self._root_mobject: SceneRootMobject = SceneRootMobject(
            background_color=ColorUtils.standardize_color(background_color)
        )
        #self._scene_ref = weakref.ref(self)
        #color_texture = Context.texture(components=3)
        #self._framebuffer: ColorFramebuffer = ColorFramebuffer(
        #    color_texture=color_texture
        #)
        #self.set_background_color(
        #    Config().style.background_color
        #)

    def _get_bound_scene(self) -> "Scene":
        return self

    async def _render(self) -> None:
        config = Toplevel.config
        fps = config.fps
        write_video = config.write_video
        write_last_frame = config.write_last_frame
        preview = config.preview
        #scene_name = cls.__name__
        #animation_dict: dict[Animation, Callable[[float], float]] = {}
        animation_dict: dict[int, Callable[[float], None] | None] = {}

        def animate(
            timestamp: float
        ) -> None:
            #for animation, absolute_rate in animation_dict.items():
            #    if (updater := animation._updater) is not None:
            #        updater(absolute_rate(timestamp))
            for composed_updater in animation_dict.values():
                if composed_updater is not None:
                    composed_updater(timestamp)

        def digest_signal(
            signal: TimelineSignal
        ) -> None:
            animation_id = signal.animation_id
            #Toplevel.set_animation_id(animation_id)
            if isinstance(signal, TimelineStartSignal):
                animation_dict[animation_id] = signal.composed_updater
            elif isinstance(signal, TimelineStopSignal):
                animation_dict.pop(animation_id)
            #match signal.timeline_state:
            #    case TimelineState.START:
            #        assert signal.animation not in animation_dict
            #        assert signal.absolute_rate is not None
            #        animation_dict[signal.animation] = signal.absolute_rate
            #    case TimelineState.STOP:
            #        animation_dict.pop(signal.animation)
            #    case TimelineState.AWAIT:
            #        pass

        async def run_frame(
            signal_timeline: Iterator[TimelineSignal],
            color_framebuffer: ColorFramebuffer,
            clock_timestamp: float,
            signal_timestamp: float,
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

            self._root_mobject._render_scene(color_framebuffer)
            if preview:
                cls._render_to_window(color_framebuffer.framebuffer)
            if video_stdin is not None:
                cls._write_frame_to_video(color_framebuffer.color_texture, video_stdin)

            return next_signal_timestamp

        async def run_frames(
            signal_timeline: Iterator[TimelineSignal],
            color_framebuffer: ColorFramebuffer,
            video_stdin: IO[bytes] | None
        ) -> None:
            spf = 1.0 / fps
            sleep_time = spf if preview else 0.0
            signal_timestamp = 0.0
            for frame_index in it.count():
                signal_timestamp, _ = await asyncio.gather(
                    run_frame(
                        signal_timeline,
                        color_framebuffer,
                        frame_index * spf,
                        signal_timestamp,
                        video_stdin
                    ),
                    asyncio.sleep(sleep_time),
                    return_exceptions=False  #True
                )
                if signal_timestamp is None:
                    break

            self._root_mobject._render_scene(color_framebuffer)
            if write_last_frame:
                cls._write_frame_to_image(color_framebuffer.color_texture)

        Context.activate(title=scene_name, standalone=not preview)
        with cls._video_writer(write_video, fps, scene_name) as video_stdin:
            self = cls()
            Toplevel.bind_animation_id_to_scene(self._id, self)
            color_framebuffer = ColorFramebuffer()
            signal_timeline = self._signal_timeline()
            await run_frames(signal_timeline, color_framebuffer, video_stdin)

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
            "-s", "{}x{}".format(*Toplevel.config.pixel_size),  # size of one frame
            "-pix_fmt", "rgb",
            "-r", str(fps),  # frames per second
            "-i", "-",  # The input comes from a pipe.
            "-vf", "vflip",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            PathUtils.output_dir.joinpath(f"{scene_name}.mp4")
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
        window = Toplevel.window._pyglet_window
        assert window is not None
        if window.is_closing:
            raise KeyboardInterrupt
        window.clear()
        assert (window_framebuffer := window.screen) is not None
        Toplevel.context.blit(framebuffer, window_framebuffer)
        window.swap_buffers()

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
        scene_name = cls.__name__
        image = Image.frombytes(
            "RGB",
            Toplevel.config.pixel_size,
            color_texture.read(),
            "raw"
        ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image.save(PathUtils.output_dir.joinpath(f"{scene_name}.png"))

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
        #for mobject in mobjects:
        #    for descendant in mobject.iter_descendants():
        #        if isinstance(descendant, RenderableMobject) and isinstance(descendant._camera_, InheritedCamera):
        #            descendant._camera_ = self._camera
        #        if isinstance(descendant, MeshMobject) and isinstance(descendant._lighting_, InheritedLighting):
        #            descendant._lighting_ = self._lighting
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
