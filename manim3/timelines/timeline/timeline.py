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
from .condition import Condition
from .conditions import Conditions
from .rate import Rate
from .rates import Rates

if TYPE_CHECKING:
    from ...toplevel.scene import Scene


class BaseAbsoluteRate:
    __slots__ = ()

    def at(
        self: Self
    ) -> float:
        return Toplevel.scene._timestamp


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


#class ScaledRate(Rate):
#    __slots__ = (
#        "_rate",
#        "_run_time_scale",
#        "_run_alpha_scale"
#    )

#    def __init__(
#        self,
#        rate: Rate,
#        run_time_scale: float,
#        run_alpha_scale: float
#    ) -> None:
#        assert run_time_scale > 0.0 and run_alpha_scale > 0.0
#        super().__init__(is_increasing=rate._is_increasing)
#        self._rate: Rate = rate
#        self._run_time_scale: float = run_time_scale
#        self._run_alpha_scale: float = run_alpha_scale

#    def at(
#        self,
#        t: float
#    ) -> float:
#        return self._rate.at(t / self._run_time_scale) * self._run_alpha_scale


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
        #"_timeline_state",
        #"_parent_absolute_rate",
        #"_relative_rate",
        #"_launch_condition",
        #"_terminate_condition",
        #"_absolute_rate",
        #"_progress_condition",
        #"_children",
        #"_construct_coroutine"
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

        # To be initialized when scheduled.
        self._schedule_info: ScheduleInfo | None = None
        self._timeline_state: TimelineState | None = None

        #self._timeline_state: TimelineState = TimelineState.UNSCHEDULED
        ## Alive in `TimelineState.BEFORE_LAUNCHED`, `TimelineState.PROGRESSING`.
        #self._parent_absolute_rate: Rate | None = None
        #self._relative_rate: Rate | None = None
        #self._launch_condition: Condition | None = None
        #self._terminate_condition: Condition | None = None
        ## Alive in `TimelineState.PROGRESSING`.
        #self._absolute_rate: Rate | None = None
        #self._progress_condition: Condition | None = None
        #self._children: list[Timeline] | None = None
        #self._construct_coroutine: Coroutine[None, None, None] | None = None

    #def submit_timeline(
    #    self: Self
    #) -> Timeline:
    #    return self

    def _schedule(
        self: Self,
        schedule_info: ScheduleInfo
        #parent_absolute_rate: Rate,
        #relative_rate: Rate,
        #launch_condition: Condition,
        #terminate_condition: Condition
    ) -> None:
        assert self._schedule_info is None
        assert self._timeline_state is None
        #schedule_info = ScheduleInfo(
        #    parent_absolute_rate=parent_absolute_rate,
        #    relative_rate=relative_rate,
        #    launch_condition=launch_condition,
        #    terminate_condition=terminate_condition
        #)
        self._schedule_info = schedule_info
        self._timeline_state = BeforeLaunched(
            launch_condition=schedule_info.launch_condition
        )
        #assert self._timeline_state == TimelineState.UNSCHEDULED
        #self._timeline_state = TimelineState.BEFORE_LAUNCHED
        #self._parent_absolute_rate = parent_absolute_rate
        #self._relative_rate = relative_rate
        #self._launch_condition = launch_condition
        #self._terminate_condition = terminate_condition

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
        #assert (schedule_info := self._schedule_info) is not None
        #assert (timeline_state := self._timeline_state) is not None
        #timeline_state = self._timeline_state
        if (timeline_state := self.get_before_launched_state()) is not None:
            if not timeline_state.launch_condition.judge():
                return
            self.launch()
            #self._timeline_state = timeline_state = OnProgressing(
            #    construct_coroutine=self.construct(),
            #    absolute_rate=AbsoluteRate(
            #        parent_absolute_rate=schedule_info.parent_absolute_rate,
            #        relative_rate=schedule_info.relative_rate
            #    ),
            #    terminate_condition=schedule_info.terminate_condition,
            #    progress_condition=Always(),
            #    children=[]
            #)
        if (timeline_state := self.get_on_progressing_state()) is not None:
            self._animation_update(timeline_state.absolute_rate.at())
            while not timeline_state.terminate_condition.judge():
                for child in timeline_state.children[:]:
                    child._progress()
                    if child.get_after_terminated_state() is not None:
                        timeline_state.children.remove(child)
                #assert (progress_condition := self._progress_condition) is not None
                if not timeline_state.progress_condition.judge():
                    return
                try:
                    timeline_state.construct_coroutine.send(None)
                except StopIteration:
                    break
            self.terminate()
            #self._timeline_state = timeline_state = AfterTerminated()
        if (timeline_state := self.get_after_terminated_state()) is not None:
            return

    #def _launch(self) -> None:
    #    assert self._timeline_state == TimelineState.BEFORE_LAUNCHED
    #    assert (parent_absolute_rate := self._parent_absolute_rate) is not None
    #    assert (relative_rate := self._relative_rate) is not None
    #    self._timeline_state = TimelineState.PROGRESSING
    #    self._construct_coroutine = self.construct()
    #    self._absolute_rate = AbsoluteRate(
    #        parent_absolute_rate=parent_absolute_rate,
    #        relative_rate=relative_rate,
    #        timestamp=Toplevel.scene._timestamp
    #    )
    #    self._progress_condition = Always()
    #    self._children = []
    #    self.update(0.0)

    #def _terminate(self) -> None:
    #    assert self._timeline_state == TimelineState.PROGRESSING
    #    self._timeline_state = TimelineState.AFTER_TERMINATED
    #    self._parent_absolute_rate = None
    #    self._relative_rate = None
    #    self._launch_condition = None
    #    self._terminate_condition = None
    #    self._construct_coroutine = None
    #    self._absolute_rate = None
    #    self._progress_condition = None
    #    self._children = None
    #    if (run_alpha := self._run_alpha) != float("inf"):
    #        self.update(run_alpha)

        #if self._timeline_state in (TimelineState.UNSCHEDULED, TimelineState.AFTER_TERMINATED):
        #    raise TypeError
        #assert (launch_condition := self._launch_condition) is not None
        #assert (terminate_condition := self._terminate_condition) is not None
        #if self._timeline_state == TimelineState.BEFORE_LAUNCHED:
        #    if not launch_condition.judge():
        #        return
        #    self._launch()
        #assert self._timeline_state == TimelineState.PROGRESSING
        #assert (absolute_rate := self._absolute_rate) is not None
        #assert (children := self._children) is not None
        #assert (construct_coroutine := self._construct_coroutine) is not None
        #self.update(absolute_rate.at(Toplevel.scene._timestamp))
        #while not terminate_condition.judge():
        #    for child in children[:]:
        #        child._progress()
        #        if child._timeline_state == TimelineState.AFTER_TERMINATED:
        #            children.remove(child)
        #    assert (progress_condition := self._progress_condition) is not None
        #    if not progress_condition.judge():
        #        return
        #    try:
        #        construct_coroutine.send(None)
        #    except StopIteration:
        #        break
        #self._terminate()

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
                #relative_rate=schedule_info.relative_rate
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
        # Must be an increasing function.
        # Forced to be `None` if `_run_alpha` is infinity.
        rate: Rate | None = None,
        # Intepreted as "the inverse of run speed" if `_run_alpha` is infinity.
        run_time: float | None = None,
        launch_condition: Condition = Conditions.always(),
        terminate_condition: Condition = Conditions.never()
    ) -> None:
        #assert self._timeline_state == TimelineState.PROGRESSING
        #assert (absolute_rate := self._absolute_rate) is not None
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
            #relative_rate=ScaledRate(
            #    rate=rate,
            #    run_time_scale=run_time_scale,
            #    run_alpha_scale=run_alpha_scale
            #),
            launch_condition=launch_condition,
            terminate_condition=terminate_condition
        ))
        #assert (children := self._children) is not None
        timeline_state.children.append(timeline)

    async def wait_until(
        self: Self,
        progress_condition: Condition
    ) -> None:
        #assert self._timeline_state == TimelineState.PROGRESSING
        #assert self._progress_condition is not None
        assert isinstance(timeline_state := self._timeline_state, OnProgressing)
        timeline_state.progress_condition = progress_condition
        await asyncio.sleep(0.0)

    # shortcuts

    @property
    def scene(
        self: Self
    ) -> Scene:
        return Toplevel.scene

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

    #async def wait_forever(self) -> None:
    #    await self.wait_until(Never())
