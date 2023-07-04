from abc import (
    ABC,
    abstractmethod
)
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterator
)
import weakref

from ..utils.rate import RateUtils

if TYPE_CHECKING:
    from ..scene.scene import Scene


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
        self._scene_ref: "weakref.ref[Scene] | None" = None

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
