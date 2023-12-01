from __future__ import annotations


import asyncio
import weakref
from abc import (
    ABC,
    abstractmethod
)
from typing import (
    TYPE_CHECKING,
    Coroutine,
    Self
)

import attrs

from ...toplevel.toplevel import Toplevel
from .conditions import (
    Condition,
    Conditions
)
from .rates import (
    Rate,
    Rates
)

if TYPE_CHECKING:
    from ...toplevel.scene import Scene


class BaseAbsoluteRate:
    __slots__ = ()

    def at(
        self: Self
    ) -> float:
        return Toplevel._get_scene()._scene_time


class AbsoluteRate(BaseAbsoluteRate):
    __slots__ = (
        "_parent_absolute_rate",
        "_rate",
        "_run_time_scale",
        "_run_alpha_scale",
        "_initial_alpha"
    )

    def __init__(
        self: Self,
        parent_absolute_rate: BaseAbsoluteRate,
        rate: Rate,
        run_time_scale: float,
        run_alpha_scale: float
    ) -> None:
        super().__init__()
        self._parent_absolute_rate: BaseAbsoluteRate = parent_absolute_rate
        self._rate: Rate = rate
        self._run_time_scale: float = run_time_scale
        self._run_alpha_scale: float = run_alpha_scale
        self._initial_alpha: float = parent_absolute_rate.at()

    def at(
        self: Self
    ) -> float:
        return self._rate.at((self._parent_absolute_rate.at() - self._initial_alpha) / self._run_time_scale) * self._run_alpha_scale


@attrs.frozen(kw_only=True)
class ScheduleInfo:
    parent_absolute_rate: BaseAbsoluteRate
    rate: Rate
    run_time_scale: float
    run_alpha_scale: float
    launch_condition: Condition
    terminate_condition: Condition


@attrs.define(kw_only=True)
class TimelineState:
    pass


@attrs.define(kw_only=True)
class BeforeLaunched(TimelineState):
    launch_condition: Condition


@attrs.define(kw_only=True)
class OnProgressing(TimelineState):
    construct_coroutine: Coroutine[None, None, None]
    absolute_rate: AbsoluteRate
    terminate_condition: Condition
    progress_condition: Condition
    children: list[Timeline]


@attrs.define(kw_only=True)
class AfterTerminated(TimelineState):
    pass


class TimelineLaunchedCondition(Condition):
    __slots__ = ("_timeline_ref",)

    def __init__(
        self: Self,
        timeline: Timeline
    ) -> None:
        super().__init__()
        self._timeline_ref: weakref.ref[Timeline] = weakref.ref(timeline)

    def judge(
        self: Self
    ) -> bool:
        timeline = self._timeline_ref()
        return (
            timeline is None
            or timeline.get_on_progressing_state() is not None
            or timeline.get_after_terminated_state() is not None
        )


class TimelineTerminatedCondition(Condition):
    __slots__ = ("_timeline_ref",)

    def __init__(
        self: Self,
        timeline: Timeline
    ) -> None:
        super().__init__()
        self._timeline_ref: weakref.ref[Timeline] = weakref.ref(timeline)

    def judge(
        self: Self
    ) -> bool:
        timeline = self._timeline_ref()
        return (
            timeline is None
            or timeline.get_after_terminated_state() is not None
        )


class TimelineProgressedDurationCondition(Condition):
    __slots__ = (
        "_timeline_ref",
        "_target_alpha"
    )

    def __init__(
        self: Self,
        timeline: Timeline,
        delta_alpha: float
    ) -> None:
        assert (timeline_state := timeline.get_on_progressing_state()) is not None
        self._timeline_ref: weakref.ref[Timeline] = weakref.ref(timeline)
        self._target_alpha: float = timeline_state.absolute_rate.at() + delta_alpha

    def judge(
        self: Self
    ) -> bool:
        timeline = self._timeline_ref()
        return (
            timeline is None
            or (timeline_state := timeline.get_on_progressing_state()) is None
            or timeline_state.absolute_rate.at() >= self._target_alpha
        )


class Timeline(ABC):
    __slots__ = (
        "__weakref__",
        "_run_alpha",
        "_schedule_info",
        "_timeline_state"
    )

    def __init__(
        self: Self,
        # The accumulated alpha value of `construct`.
        # Left as `inf` if infinite or indefinite.
        # This parameter is required mostly for the program to know
        # how long the timeline is before running the timeline.
        run_alpha: float = float("inf")
    ) -> None:
        super().__init__()
        self._run_alpha: float = run_alpha
        self._schedule_info: ScheduleInfo | None = None
        self._timeline_state: TimelineState | None = None

    def _schedule(
        self: Self,
        schedule_info: ScheduleInfo
    ) -> None:
        assert self._schedule_info is None
        assert self._timeline_state is None
        self._schedule_info = schedule_info
        self._timeline_state = BeforeLaunched(
            launch_condition=schedule_info.launch_condition
        )

    def _root_schedule(
        self: Self
    ) -> None:
        self._schedule(ScheduleInfo(
            parent_absolute_rate=BaseAbsoluteRate(),
            rate=Rates.linear(),
            run_time_scale=1.0,
            run_alpha_scale=1.0,
            launch_condition=Conditions.always(),
            terminate_condition=Conditions.never()
        ))

    def _progress(
        self: Self
    ) -> None:
        if (timeline_state := self.get_before_launched_state()) is not None:
            if not timeline_state.launch_condition.judge():
                return
            self.launch()
        if (timeline_state := self.get_on_progressing_state()) is not None:
            self._animation_update(timeline_state.absolute_rate.at())
            while not timeline_state.terminate_condition.judge():
                for child in timeline_state.children[:]:
                    child._progress()
                    if child.get_after_terminated_state() is not None:
                        timeline_state.children.remove(child)
                if not timeline_state.progress_condition.judge():
                    return
                try:
                    timeline_state.construct_coroutine.send(None)
                except StopIteration:
                    break
            self.terminate()
        if (timeline_state := self.get_after_terminated_state()) is not None:
            return

    def _animation_update(
        self: Self,
        time: float
    ) -> None:
        pass

    @abstractmethod
    async def construct(
        self: Self
    ) -> None:
        pass

    def get_before_launched_state(
        self: Self
    ) -> BeforeLaunched | None:
        if isinstance(timeline_state := self._timeline_state, BeforeLaunched):
            return timeline_state
        return None

    def get_on_progressing_state(
        self: Self
    ) -> OnProgressing | None:
        if isinstance(timeline_state := self._timeline_state, OnProgressing):
            return timeline_state
        return None

    def get_after_terminated_state(
        self: Self
    ) -> AfterTerminated | None:
        if isinstance(timeline_state := self._timeline_state, AfterTerminated):
            return timeline_state
        return None

    def launch(
        self: Self
    ) -> None:
        assert self.get_before_launched_state() is not None
        assert (schedule_info := self._schedule_info) is not None
        self._timeline_state = OnProgressing(
            construct_coroutine=self.construct(),
            absolute_rate=AbsoluteRate(
                parent_absolute_rate=schedule_info.parent_absolute_rate,
                rate=schedule_info.rate,
                run_time_scale=schedule_info.run_time_scale,
                run_alpha_scale=schedule_info.run_alpha_scale
            ),
            terminate_condition=schedule_info.terminate_condition,
            progress_condition=Conditions.always(),
            children=[]
        )

    def terminate(
        self: Self
    ) -> None:
        assert self.get_on_progressing_state() is not None
        assert self._schedule_info is not None
        self._timeline_state = AfterTerminated()

    def prepare(
        self: Self,
        timeline: Timeline,
        *,
        # `[0.0, +infty) -> [0.0, +infty), time |-> alpha`
        # `_is_increasing_` must be true.
        # Must be `None` if `_run_alpha` is infinity.
        rate: Rate | None = None,
        # Intepreted as "the inverse of run speed" if `_run_alpha` is infinity.
        run_time: float | None = None,
        launch_condition: Condition = Conditions.always(),
        terminate_condition: Condition = Conditions.never()
    ) -> None:
        assert isinstance(timeline_state := self._timeline_state, OnProgressing)
        if (run_alpha := timeline._run_alpha) == float("inf"):
            assert rate is None
            run_alpha_scale = 1.0
        else:
            run_alpha_scale = run_alpha
        if run_time is None:
            run_time_scale = run_alpha_scale
        else:
            run_time_scale = run_time
        if rate is None:
            rate = Rates.linear()
        assert rate._is_increasing_
        timeline._schedule(ScheduleInfo(
            parent_absolute_rate=timeline_state.absolute_rate,
            rate=rate,
            run_time_scale=run_time_scale,
            run_alpha_scale=run_alpha_scale,
            launch_condition=launch_condition,
            terminate_condition=terminate_condition
        ))
        timeline_state.children.append(timeline)

    async def wait_until(
        self: Self,
        progress_condition: Condition
    ) -> None:
        assert isinstance(timeline_state := self._timeline_state, OnProgressing)
        timeline_state.progress_condition = progress_condition
        await asyncio.sleep(0.0)

    # shortcuts

    @property
    def scene(
        self: Self
    ) -> Scene:
        return Toplevel._get_scene()

    async def play(
        self: Self,
        timeline: Timeline,
        rate: Rate | None = None,
        run_time: float | None = None,
        launch_condition: Condition = Conditions.always(),
        terminate_condition: Condition = Conditions.never()
    ) -> None:
        self.prepare(
            timeline,
            rate=rate,
            run_time=run_time,
            launch_condition=launch_condition,
            terminate_condition=terminate_condition
        )
        await self.wait_until(timeline.terminated())

    async def wait(
        self: Self,
        delta_alpha: float = 1.0
    ) -> None:
        await self.wait_until(self.progressed_duration(delta_alpha))

    # conditions

    def launched(
        self: Self
    ) -> TimelineLaunchedCondition:
        return TimelineLaunchedCondition(self)

    def terminated(
        self: Self
    ) -> TimelineTerminatedCondition:
        return TimelineTerminatedCondition(self)

    def progressed_duration(
        self: Self,
        delta_alpha: float
    ) -> TimelineProgressedDurationCondition:
        return TimelineProgressedDurationCondition(self, delta_alpha)
