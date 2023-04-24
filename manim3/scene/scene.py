#import asyncio
#from typing import Coroutine

#import numpy as np

#from typing import Generator
from ..animations.animation import (
    Animation,
    UpdaterItem,
    UpdaterItemVerb
)
from ..custom_typing import (
    ColorT,
    Vec3T
)
from ..mobjects.mobject import Mobject
from ..mobjects.scene_frame import SceneFrame
from ..rendering.config import (
    Config,
    ConfigSingleton
)
from ..rendering.context import Context
from ..scene.scene_state import SceneState


class EndSceneException(Exception):
    pass


class Scene(Animation):
    __slots__ = (
        "_scene_state",
        "_scene_frame"
        #"_construct_coroutines",
        #"_animation_dict",
        #"_frame_floating_index",
        #"_previous_frame_rendering_timestamp"
    )

    def __init__(self) -> None:
        super().__init__(
            start_time=ConfigSingleton().rendering.start_time,
            run_time=ConfigSingleton().rendering.run_time
        )
        self._scene_state: SceneState = SceneState()
        self._scene_frame: SceneFrame = SceneFrame()
        #self._construct_coroutines: list[Coroutine] = [self.construct()]
        #self._animation_dict: dict[Animation, float] = {}
        # A timer scaled by fps.
        #self._frame_floating_index: float = 0.0
        #self._previous_frame_rendering_timestamp: float | None = None

    #@classmethod
    #def _find_frame_range(
    #    cls,
    #    start_frame_floating_index: float,
    #    stop_frame_floating_index: float
    #) -> tuple[range, bool]:
    #    # Find all frame indices in the intersection of
    #    # `(start_frame_floating_index, stop_frame_floating_index]`
    #    # and `[ConfigSingleton().start_frame_index, ConfigSingleton().stop_frame_index]`.
    #    config_start_frame_index = ConfigSingleton().rendering.start_frame_index
    #    config_stop_frame_index = ConfigSingleton().rendering.stop_frame_index
    #    start_frame_index = int(np.ceil(
    #        start_frame_floating_index
    #        if config_start_frame_index is None
    #        else max(config_start_frame_index, start_frame_floating_index)
    #    ))
    #    stop_frame_index = int(np.floor(
    #        stop_frame_floating_index
    #        if config_stop_frame_index is None
    #        else min(config_stop_frame_index, stop_frame_floating_index)
    #    ))
    #    if np.isclose(start_frame_index, start_frame_floating_index):
    #        # Exclude the open side.
    #        start_frame_index += 1
    #    reaches_end = config_stop_frame_index is not None and bool(np.isclose(stop_frame_index, config_stop_frame_index))
    #    return range(start_frame_index, stop_frame_index + 1), reaches_end

    #@classmethod
    #def _regroup(
    #    cls,
    #    regroup_item: RegroupItem
    #) -> None:
    #    mobjects = regroup_item.mobjects
    #    if isinstance(mobjects, Mobject):
    #        mobjects = (mobjects,)
    #    targets = regroup_item.targets
    #    if isinstance(targets, Mobject):
    #        targets = (targets,)
    #    for mobject in dict.fromkeys(mobjects):
    #        if regroup_item.verb == RegroupVerb.ADD:
    #            mobject.add(*targets)
    #        elif regroup_item.verb == RegroupVerb.BECOMES:
    #            mobject.becomes(next(iter(targets)))
    #        elif regroup_item.verb == RegroupVerb.DISCARD:
    #            mobject.discard(*targets)

    #def _update_dt(
    #    self,
    #    dt: float
    #):
    #    assert dt >= 0.0
    #    for animation in list(self._animation_dict):
    #        t0 = self._animation_dict[animation]
    #        t = t0 + dt
    #        self._animation_dict[animation] = t
    #        if t < animation._start_time:
    #            continue

    #        animation_expired = False
    #        if (stop_time := animation._stop_time) is not None and t > stop_time:
    #            animation_expired = True
    #            t = stop_time

    #        for time_regroup_item in animation._time_regroup_items[:]:
    #            regroup_time, regroup_item = time_regroup_item
    #            if t < regroup_time:
    #                continue
    #            self._regroup(regroup_item)
    #            animation._time_regroup_items.remove(time_regroup_item)

    #        animation._time_animate_func(t0, t)

    #        if animation_expired:
    #            assert not animation._time_regroup_items
    #            self._animation_dict.pop(animation)

    #    return self

    #def _update_frames(
    #    self,
    #    frames: float
    #):
    #    self._update_dt(frames / ConfigSingleton().rendering.fps)
    #    return self

    def _run(self) -> None:
        #animations = list(self._iter_animation_descendants())
        #while animations:
        #    for animation in animations:
        #        if animation._rest_wait_time is not None:
        #            continue
        #        animation._coroutine.send(None)

        def update(
            timestamp: float,
            updater_items: list[UpdaterItem]
        ) -> None:
            for updater_item in updater_items:
                if (updater := updater_item.updater) is None:
                    continue
                updater(updater_item.absolute_rate_func(timestamp))

        #timeline = self._accumulated_timeline()
        #prev_timestamp = self._lag_time
        spf = 1.0 / ConfigSingleton().rendering.fps
        timestamp = 0.0
        updater_items: list[UpdaterItem] = []
        for state in self._accumulated_timeline():
            #try:
            #    state = timeline.send(None)
            #except StopIteration:
            #    break

            if state.verb == UpdaterItemVerb.APPEND:
                updater_items.append(state.updater_item)
            elif state.verb == UpdaterItemVerb.REMOVE:
                updater_items.remove(state.updater_item)

            final_timestamp = state.timestamp
            while timestamp < final_timestamp:
                update(timestamp, updater_items)
                self._scene_frame._process_rendering(render_to_video=True)
                timestamp += spf
            update(final_timestamp, updater_items)

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

    #def prepare(
    #    self,
    #    *animations: Animation
    #):
    #    for animation in animations:
    #        self._animation_dict[animation] = 0.0
    #    return self

    #async def wait(
    #    self,
    #    t: float = 1.0
    #) -> None:
    #    assert t >= 0.0
    #    frames = t * ConfigSingleton().rendering.fps
    #    start_frame_floating_index = self._frame_floating_index
    #    stop_frame_floating_index = start_frame_floating_index + frames
    #    self._frame_floating_index = stop_frame_floating_index
    #    frame_range, reaches_end = self._find_frame_range(start_frame_floating_index, stop_frame_floating_index)

    #    if not frame_range:
    #        self._update_frames(frames)
    #    else:
    #        self._update_frames(frame_range.start - start_frame_floating_index)
    #        if self._previous_frame_rendering_timestamp is None:
    #            self._process_rendering(render_to_video=True)
    #        for _ in frame_range[:-1]:
    #            self._update_frames(1)
    #            self._process_rendering(render_to_video=True)
    #        self._update_frames(stop_frame_floating_index - (frame_range.stop - 1))

    #    if reaches_end:
    #        raise EndSceneException()

    #async def play(
    #    self,
    #    *animations: Animation
    #) -> None:
    #    self.prepare(*animations)
    #    try:
    #        wait_time = max(t for animation in animations if (t := animation._stop_time) is not None)
    #    except ValueError:
    #        wait_time = 0.0
    #    await self.wait(wait_time)

    #async def construct(self) -> None:
    #    await asyncio.sleep(0.0)

    def set_view(
        self,
        *,
        eye: Vec3T | None = None,
        target: Vec3T | None = None,
        up: Vec3T | None = None
    ):
        self._scene_state.set_view(
            eye=eye,
            target=target,
            up=up
        )
        return self

    def set_background(
        self,
        *,
        color: ColorT | None = None,
        opacity: float | None = None
    ):
        self._scene_state.set_background(
            color=color,
            opacity=opacity
        )
        return self

    def set_ambient_light(
        self,
        *,
        color: ColorT | None = None,
        opacity: float | None = None
    ):
        self._scene_state.set_ambient_light(
            color=color,
            opacity=opacity
        )
        return self

    def add_point_light(
        self,
        *,
        position: Vec3T | None = None,
        color: ColorT | None = None,
        opacity: float | None = None
    ):
        self._scene_state.add_point_light(
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
        color: ColorT | None = None,
        opacity: float | None = None
    ):
        self._scene_state.set_point_light(
            index=index,
            position=position,
            color=color,
            opacity=opacity
        )
        return self

    def set_style(
        self,
        *,
        background_color: ColorT | None = None,
        background_opacity: float | None = None,
        ambient_light_color: ColorT | None = None,
        ambient_light_opacity: float | None = None,
        point_light_position: Vec3T | None = None,
        point_light_color: ColorT | None = None,
        point_light_opacity: float | None = None
    ):
        self._scene_state.set_style(
            background_color=background_color,
            background_opacity=background_opacity,
            ambient_light_color=ambient_light_color,
            ambient_light_opacity=ambient_light_opacity,
            point_light_position=point_light_position,
            point_light_color=point_light_color,
            point_light_opacity=point_light_opacity
        )
        return self

    @classmethod
    def render(
        cls,
        config: Config | None = None
    ) -> None:
        if config is None:
            config = Config()

        ConfigSingleton.set(config)
        Context.activate()
        if ConfigSingleton().rendering.write_video:
            Context.setup_writing_process(cls.__name__)

        self = cls()

        try:
            try:
                self._run()
            except EndSceneException:
                pass
            finally:
                if ConfigSingleton().rendering.write_last_frame:
                    self._scene_frame._process_rendering(render_to_image=True)
        except KeyboardInterrupt:
            pass
        finally:
            if ConfigSingleton().rendering.write_video:
                writing_process = Context.writing_process
                assert writing_process.stdin is not None
                writing_process.stdin.close()
                writing_process.wait()
                writing_process.terminate()
