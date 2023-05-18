from abc import ABC
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Iterator
)
import weakref

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
from ..mobjects.renderable_mobject import RenderableMobject
from ..mobjects.scene_frame import SceneFrame
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

    async def timeline(self) -> None:
        # Wait forever. Default timeline for infinite animation.
        await self.wait(600)


class Scene(Animation):
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
        self._scene_ref = weakref.ref(self)

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
            color_texture: moderngl.Texture
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

            self._render_to_texture(color_texture)
            if ConfigSingleton().rendering.preview:
                self._scene_frame._render_to_window(color_texture)
            if ConfigSingleton().rendering.write_video:
                self._write_to_writing_process(color_texture)

            return next_signal_timestamp

        async def run_frames(
            color_texture: moderngl.Texture
        ) -> None:
            signal_timestamp = 0.0
            for clock_timestamp, sleep_time in frame_clock(
                fps=ConfigSingleton().rendering.fps,
                sleep=ConfigSingleton().rendering.preview
            ):
                signal_timestamp, _ = await asyncio.gather(
                    run_frame(
                        clock_timestamp,
                        signal_timestamp,
                        color_texture
                    ),
                    asyncio.sleep(sleep_time)
                )
                if signal_timestamp is None:
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
        for mobject in scene_frame.iter_descendants_by_type(mobject_type=RenderableMobject):
            mobject._camera_uniform_block_buffer_ = camera._camera_uniform_block_buffer_

        lighting = self._lighting
        lighting._ambient_lights_ = scene_frame.iter_descendants_by_type(mobject_type=AmbientLight)
        lighting._point_lights_ = scene_frame.iter_descendants_by_type(mobject_type=PointLight)
        for mobject in scene_frame.iter_descendants_by_type(mobject_type=MeshMobject):
            mobject._lighting_uniform_block_buffer_ = lighting._lighting_uniform_block_buffer_

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
