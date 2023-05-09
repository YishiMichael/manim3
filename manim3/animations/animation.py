from dataclasses import dataclass
from typing import (
    Callable,
    Iterator
)

from ..custom_typing import TimelineT
from ..utils.rate import RateUtils


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class UpdaterItem:
    updater: Callable[[float], None]
    absolute_rate: Callable[[float], float]


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class UpdaterItemAppendSignal:
    updater_item: UpdaterItem


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class UpdaterItemRemoveSignal:
    updater_item: UpdaterItem


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class AwaitSignal:
    pass


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TimelineState:
    timestamp: float
    signal: UpdaterItemAppendSignal | UpdaterItemRemoveSignal | AwaitSignal


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


class Animation:
    __slots__ = (
        "_run_time",
        "_relative_rate",
        "_updater",
        "_new_children"
    )

    def __init__(
        self,
        # If provided, the animation will be clipped from right at `run_time`.
        # `parent.play(self)` will call `parent.wait(run_time)` that covers this animation.
        run_time: float | None = None,
        # `[0.0, run_time] -> [0.0, +infty), time |-> alpha`
        # Must be an increasing function.
        relative_rate: Callable[[float], float] = RateUtils.linear,
        # Update continuously (per frame).
        updater: Callable[[float], None] | None = None
    ) -> None:
        super().__init__()
        assert run_time is None or run_time >= 0.0
        self._run_time: float | None = run_time
        self._relative_rate: Callable[[float], float] = relative_rate
        self._updater: Callable[[float], None] | None = updater
        self._new_children: list[Animation] = []

    # Yield `delta_alpha` values.
    def timeline(self) -> TimelineT:
        yield from self.wait(1024)  # Wait forever...

    def _absolute_timeline(self) -> Iterator[TimelineState]:
        relative_rate = self._relative_rate
        relative_rate_inv = RateUtils.inverse(relative_rate)
        run_alpha = relative_rate(self._run_time) if self._run_time is not None else None
        current_alpha = relative_rate(0.0)
        assert current_alpha >= 0.0

        manager = TimelineManager()
        updater_item_convert_dict: dict[UpdaterItem, UpdaterItem] = {}

        if self._updater is not None:
            self_updater_item = UpdaterItem(
                updater=self._updater,
                absolute_rate=relative_rate
            )
        else:
            self_updater_item = None

        if self_updater_item is not None:
            signal = UpdaterItemAppendSignal(
                updater_item=self_updater_item
            )
        else:
            signal = AwaitSignal()
        yield TimelineState(
            timestamp=0.0,
            signal=signal
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
                if isinstance(signal, UpdaterItemAppendSignal):
                    updater_item = UpdaterItem(
                        updater=signal.updater_item.updater,
                        absolute_rate=RateUtils.compose(
                            RateUtils.adjust(signal.updater_item.absolute_rate, lag_time=timeline_start_alpha),
                            relative_rate
                        )
                    )
                    updater_item_convert_dict[signal.updater_item] = updater_item
                    new_signal = UpdaterItemAppendSignal(
                        updater_item=updater_item
                    )
                elif isinstance(signal, UpdaterItemRemoveSignal):
                    updater_item = updater_item_convert_dict.pop(signal.updater_item)
                    new_signal = UpdaterItemRemoveSignal(
                        updater_item=updater_item
                    )
                elif isinstance(signal, AwaitSignal):
                    new_signal = AwaitSignal()
                else:
                    raise TypeError

                yield TimelineState(
                    timestamp=relative_rate_inv(next_alpha),
                    signal=new_signal
                )
                manager.advance_state(timeline)

            yield TimelineState(
                timestamp=relative_rate_inv(current_alpha),
                signal=AwaitSignal()
            )

            if early_break:
                break

        if self_updater_item is not None:
            signal = UpdaterItemRemoveSignal(
                updater_item=self_updater_item
            )
        else:
            signal = AwaitSignal()
        yield TimelineState(
            timestamp=relative_rate_inv(current_alpha),
            signal=signal
        )

    def prepare(
        self,
        *animations: "Animation"
    ) -> None:
        self._new_children.extend(animations)

    def wait(
        self,
        delta_alpha: float = 1.0
    ) -> TimelineT:
        yield delta_alpha

    def play(
        self,
        *animations: "Animation"
    ) -> TimelineT:
        self.prepare(*animations)
        delta_alpha = max((
            run_time
            for animation in animations
            if (run_time := animation._run_time) is not None
        ), default=0.0)
        yield from self.wait(delta_alpha)
