import time
from typing import Iterator

from ..animations.animation import (
    Animation,
    AwaitSignal,
    UpdaterItem,
    UpdaterItemAppendSignal,
    UpdaterItemRemoveSignal
)
from ..mobjects.mobject import Mobject
from ..mobjects.scene_frame import SceneFrame
from ..rendering.config import (
    Config,
    ConfigSingleton
)
from ..rendering.context import Context
from ..scene.scene_state import SceneState
from ..utils.rate import RateUtils


class Scene(Animation):
    __slots__ = (
        "_scene_state",
        "_scene_frame"
    )

    def __init__(self) -> None:
        start_time = ConfigSingleton().rendering.start_time
        stop_time = ConfigSingleton().rendering.stop_time
        super().__init__(
            run_time=stop_time - start_time if stop_time is not None else None,
            relative_rate=RateUtils.adjust(RateUtils.linear, lag_time=-start_time)
        )
        self._scene_state: SceneState = SceneState()
        self._scene_frame: SceneFrame = SceneFrame()

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

        def update(
            timestamp: float,
            updater_items: list[UpdaterItem]
        ) -> None:
            for updater_item in updater_items:
                updater_item.updater(updater_item.absolute_rate(timestamp))

        absolute_timeline = self._absolute_timeline()
        state_final_timestamp = 0.0
        updater_items: list[UpdaterItem] = []
        terminated: bool = False
        for timestamp in frame_clock(
            fps=ConfigSingleton().rendering.fps,
            sleep=ConfigSingleton().rendering.preview
        ):
            while state_final_timestamp <= timestamp:
                update(state_final_timestamp, updater_items)
                try:
                    state = next(absolute_timeline)
                except StopIteration:
                    terminated = True
                    break
                state_final_timestamp = state.timestamp
                signal = state.signal
                if isinstance(signal, UpdaterItemAppendSignal):
                    updater_items.append(signal.updater_item)
                elif isinstance(signal, UpdaterItemRemoveSignal):
                    updater_items.remove(signal.updater_item)
                elif isinstance(signal, AwaitSignal):
                    pass
                else:
                    raise TypeError
            if terminated:
                break
            update(timestamp, updater_items)
            self._scene_frame._process_rendering(
                scene_state=self._scene_state,
                render_to_video=True
            )

        if ConfigSingleton().rendering.write_last_frame:
            self._scene_frame._process_rendering(
                scene_state=self._scene_state,
                render_to_image=True
            )

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

    @property
    def scene_state(self) -> SceneState:
        return self._scene_state

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
                writing_process = Context.writing_process
                assert writing_process.stdin is not None
                writing_process.stdin.close()
                writing_process.wait()
                writing_process.terminate()
