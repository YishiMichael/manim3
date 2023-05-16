from abc import ABC
import asyncio
from dataclasses import dataclass
from typing import (
    Callable,
    Iterator
)

from ..utils.rate import RateUtils


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TimelineItem:
    updater: Callable[[float], None] | None
    absolute_rate: Callable[[float], float]


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TimelineItemAppendSignal:
    timeline_item: TimelineItem


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TimelineItemRemoveSignal:
    timeline_item: TimelineItem


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TimelineAwaitSignal:
    pass


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TimelineState:
    timestamp: float
    signal: TimelineItemAppendSignal | TimelineItemRemoveSignal | TimelineAwaitSignal


class TimelineManager:
    __slots__ = (
        "start_alpha_dict",
        "state_dict"
    )

    def __init__(self) -> None:
        super().__init__()
        self.start_alpha_dict: dict[Iterator[TimelineState], float] = {}
        self.state_dict: dict[Iterator[TimelineState], TimelineState] = {}

    def add_state_timeline(
        self,
        state_timeline: Iterator[TimelineState],
        start_alpha: float
    ) -> None:
        try:
            state = next(state_timeline)
        except StopIteration:
            return
        self.start_alpha_dict[state_timeline] = start_alpha
        self.state_dict[state_timeline] = state

    def advance_state(
        self,
        state_timeline: Iterator[TimelineState]
    ) -> None:
        try:
            state = next(state_timeline)
        except StopIteration:
            self.start_alpha_dict.pop(state_timeline)
            self.state_dict.pop(state_timeline)
            return
        self.state_dict[state_timeline] = state

    def is_not_empty(self) -> bool:
        return bool(self.state_dict)

    def get_next_state_timeline_item(self) -> tuple[Iterator[TimelineState], float, TimelineState]:
        start_alpha_dict = self.start_alpha_dict
        state_dict = self.state_dict

        def get_next_alpha(
            state_timeline: Iterator[TimelineState]
        ) -> float:
            next_alpha = start_alpha_dict[state_timeline] + state_dict[state_timeline].timestamp
            return round(next_alpha, 3)  # Avoid floating error.

        state_timeline = min(state_dict, key=get_next_alpha)
        return state_timeline, start_alpha_dict[state_timeline], state_dict[state_timeline]


class Timeline(ABC):
    __slots__ = (
        "_updater",
        "_run_time",
        "_relative_rate",
        "_delta_alpha",
        "_new_children"
    )

    def __init__(
        self,
        # Update continuously (per frame).
        updater: Callable[[float], None] | None = None,
        # If provided, the timeline will be clipped from right at `run_time`.
        # `parent.play(self)` will call `parent.wait(run_time)` that covers this timeline.
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
        self._new_children: list[Timeline] = []

    def _wait_timeline(self) -> Iterator[float]:
        timeline_coroutine = self.timeline()
        try:
            while True:
                timeline_coroutine.send(None)
                yield self._delta_alpha
        except StopIteration:
            pass

    def _state_timeline(self) -> Iterator[TimelineState]:
        relative_rate = self._relative_rate
        relative_rate_inv = RateUtils.inverse(relative_rate)
        run_alpha = relative_rate(self._run_time) if self._run_time is not None else None
        current_alpha = relative_rate(0.0)
        assert current_alpha >= 0.0

        manager = TimelineManager()
        timeline_item_convert_dict: dict[TimelineItem, TimelineItem] = {}

        self_timeline_item = TimelineItem(
            updater=self._updater,
            absolute_rate=relative_rate
        )
        yield TimelineState(
            timestamp=0.0,
            signal=TimelineItemAppendSignal(
                timeline_item=self_timeline_item
            )
        )

        for wait_delta_alpha in self._wait_timeline():
            for child in self._new_children:
                manager.add_state_timeline(
                    state_timeline=child._state_timeline(),
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
                state_timeline, timeline_start_alpha, state = manager.get_next_state_timeline_item()
                next_alpha = timeline_start_alpha + state.timestamp
                if next_alpha > current_alpha:
                    break

                match state.signal:
                    case TimelineItemAppendSignal(timeline_item=timeline_item):
                        new_timeline_item = TimelineItem(
                            updater=timeline_item.updater,
                            absolute_rate=RateUtils.compose(
                                RateUtils.adjust(timeline_item.absolute_rate, lag_time=timeline_start_alpha),
                                relative_rate
                            )
                        )
                        timeline_item_convert_dict[timeline_item] = new_timeline_item
                        new_signal = TimelineItemAppendSignal(
                            timeline_item=new_timeline_item
                        )
                    case TimelineItemRemoveSignal(timeline_item=timeline_item):
                        new_timeline_item = timeline_item_convert_dict.pop(timeline_item)
                        new_signal = TimelineItemRemoveSignal(
                            timeline_item=new_timeline_item
                        )
                    case TimelineAwaitSignal():
                        new_signal = TimelineAwaitSignal()

                yield TimelineState(
                    timestamp=relative_rate_inv(next_alpha),
                    signal=new_signal
                )
                manager.advance_state(state_timeline)

            yield TimelineState(
                timestamp=relative_rate_inv(current_alpha),
                signal=TimelineAwaitSignal()
            )

            if early_break:
                break

        yield TimelineState(
            timestamp=relative_rate_inv(current_alpha),
            signal=TimelineItemRemoveSignal(
                timeline_item=self_timeline_item
            )
        )

    def _is_prepared(
        self,
        timeline: "Timeline"
    ) -> None:
        # Handle `prepare` from the other direction. Implemented in subclasses.
        pass

    def prepare(
        self,
        *timelines: "Timeline"
    ) -> None:
        for timeline in timelines:
            timeline._is_prepared(self)
        self._new_children.extend(timelines)

    async def wait(
        self,
        delta_alpha: float = 1.0
    ) -> None:
        self._delta_alpha = delta_alpha
        await asyncio.sleep(0.0)

    async def play(
        self,
        *timelines: "Timeline"
    ) -> None:
        self.prepare(*timelines)
        delta_alpha = max((
            run_time
            for timeline in timelines
            if (run_time := timeline._run_time) is not None
        ), default=0.0)
        await self.wait(delta_alpha)

    async def timeline(self) -> None:
        await self.wait(1024)  # Wait forever...
