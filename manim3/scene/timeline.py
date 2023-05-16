from dataclasses import dataclass
from typing import (
    Callable,
    Iterator
)

from ..custom_typing import TimelineReturnT
from ..utils.rate import RateUtils


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TimelineItem:
    timeline: "Timeline"
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
        "timeline_start_alpha_dict",
        "state_dict"
    )

    def __init__(self) -> None:
        super().__init__()
        self.timeline_start_alpha_dict: dict[Iterator[TimelineState], float] = {}
        self.state_dict: dict[Iterator[TimelineState], TimelineState] = {}

    def add_timeline(
        self,
        timeline: Iterator[TimelineState],
        start_alpha: float
    ) -> None:
        try:
            state = next(timeline)
        except StopIteration:
            return
        self.timeline_start_alpha_dict[timeline] = start_alpha
        self.state_dict[timeline] = state

    def advance_state(
        self,
        timeline: Iterator[TimelineState]
    ) -> None:
        try:
            state = next(timeline)
        except StopIteration:
            self.timeline_start_alpha_dict.pop(timeline)
            self.state_dict.pop(timeline)
            return
        self.state_dict[timeline] = state

    def is_not_empty(self) -> bool:
        return bool(self.state_dict)

    def get_next_timeline_item(self) -> tuple[Iterator[TimelineState], float, TimelineState]:
        timeline_start_alpha_dict = self.timeline_start_alpha_dict
        state_dict = self.state_dict

        def get_next_alpha(
            timeline: Iterator[TimelineState]
        ) -> float:
            next_alpha = timeline_start_alpha_dict[timeline] + state_dict[timeline].timestamp
            return round(next_alpha, 3)  # Avoid floating error.

        timeline = min(state_dict, key=get_next_alpha)
        return timeline, timeline_start_alpha_dict[timeline], state_dict[timeline]


#class TimelineItems:
#    __slots__ = ("_items",)

#    _instance: "ClassVar[TimelineItems | None]" = None

#    def __init__(self) -> None:
#        super().__init__()
#        self._items: list[TimelineItem] = []

#    def __enter__(self):
#        cls = type(self)
#        assert cls._instance is None
#        cls._instance = self
#        return self

#    def __exit__(
#        self,
#        exc_type,
#        exc_val,
#        exc_tb
#    ) -> None:
#        cls = type(self)
#        assert cls._instance is self
#        cls._instance = None

#    def digest_signal(
#        self,
#        signal: TimelineItemAppendSignal | TimelineItemRemoveSignal | TimelineAwaitSignal
#    ) -> None:
#        if isinstance(signal, TimelineItemAppendSignal):
#            self._items.append(signal.timeline_item)
#        elif isinstance(signal, TimelineItemRemoveSignal):
#            self._items.remove(signal.timeline_item)

#    def animate(
#        self,
#        timestamp: float
#    ) -> None:
#        for timeline_item in self._items:
#            if (updater := timeline_item.timeline._updater) is not None:
#                updater(timeline_item.absolute_rate(timestamp))

#    @classmethod
#    def current_scene(cls) -> "Scene":
#        assert (self := cls._instance) is not None
#        for timeline_item in reversed(self._items):
#            if isinstance(timeline := timeline_item.timeline, Scene):
#                return timeline
#        raise ValueError


class Timeline:
    __slots__ = (
        "_updater",
        "_run_time",
        "_relative_rate",
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
        self._new_children: list[Timeline] = []

    def _absolute_timeline(self) -> Iterator[TimelineState]:
        relative_rate = self._relative_rate
        relative_rate_inv = RateUtils.inverse(relative_rate)
        run_alpha = relative_rate(self._run_time) if self._run_time is not None else None
        current_alpha = relative_rate(0.0)
        assert current_alpha >= 0.0

        manager = TimelineManager()
        timeline_item_convert_dict: dict[TimelineItem, TimelineItem] = {}

        #if self._updater is not None:
        #    self_updater_item = UpdaterItem(
        #        updater=self._updater,
        #        absolute_rate=relative_rate
        #    )
        #else:
        #    self_updater_item = None
        self_timeline_item = TimelineItem(
            timeline=self,
            absolute_rate=relative_rate
        )

        #if self_updater_item is not None:
        #    signal = UpdaterItemAppendSignal(
        #        updater_item=self_updater_item
        #    )
        #else:
        #    signal = TimelineAwaitSignal()
        yield TimelineState(
            timestamp=0.0,
            signal=TimelineItemAppendSignal(
                timeline_item=self_timeline_item
            )
        )

        for wait_delta_alpha in self.timeline():
            for child in self._new_children:
                manager.add_timeline(
                    timeline=child._absolute_timeline(),
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
                timeline, timeline_start_alpha, state = manager.get_next_timeline_item()
                next_alpha = timeline_start_alpha + state.timestamp
                if next_alpha > current_alpha:
                    break

                signal = state.signal
                if isinstance(signal, TimelineItemAppendSignal):
                    timeline_item = TimelineItem(
                        timeline=signal.timeline_item.timeline,
                        absolute_rate=RateUtils.compose(
                            RateUtils.adjust(signal.timeline_item.absolute_rate, lag_time=timeline_start_alpha),
                            relative_rate
                        )
                    )
                    timeline_item_convert_dict[signal.timeline_item] = timeline_item
                    new_signal = TimelineItemAppendSignal(
                        timeline_item=timeline_item
                    )
                elif isinstance(signal, TimelineItemRemoveSignal):
                    timeline_item = timeline_item_convert_dict.pop(signal.timeline_item)
                    new_signal = TimelineItemRemoveSignal(
                        timeline_item=timeline_item
                    )
                elif isinstance(signal, TimelineAwaitSignal):
                    new_signal = TimelineAwaitSignal()
                else:
                    raise TypeError

                yield TimelineState(
                    timestamp=relative_rate_inv(next_alpha),
                    signal=new_signal
                )
                manager.advance_state(timeline)

            yield TimelineState(
                timestamp=relative_rate_inv(current_alpha),
                signal=TimelineAwaitSignal()
            )

            if early_break:
                break

        #if self_updater_item is not None:
        #    signal = UpdaterItemRemoveSignal(
        #        updater_item=self_updater_item
        #    )
        #else:
        #    signal = TimelineAwaitSignal()
        yield TimelineState(
            timestamp=relative_rate_inv(current_alpha),
            signal=TimelineItemRemoveSignal(
                timeline_item=self_timeline_item
            )
        )

    # Yield `delta_alpha` values.
    def timeline(self) -> TimelineReturnT:
        yield from self.wait(1024)  # Wait forever...

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

    def wait(
        self,
        delta_alpha: float = 1.0
    ) -> TimelineReturnT:
        yield delta_alpha

    def play(
        self,
        *timelines: "Timeline"
    ) -> TimelineReturnT:
        self.prepare(*timelines)
        delta_alpha = max((
            run_time
            for timeline in timelines
            if (run_time := timeline._run_time) is not None
        ), default=0.0)
        yield from self.wait(delta_alpha)
