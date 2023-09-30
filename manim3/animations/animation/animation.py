import asyncio
import weakref
from abc import (
    ABC,
    abstractmethod
)
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Coroutine
)

from ...toplevel.toplevel import Toplevel
#from .animation_state import AnimationState
#from .conditions.always import Always
from .condition import Condition
from .conditions import Conditions
#from .conditions.never import Never
#from .conditions.progress_duration import ProgressDuration
#from .conditions.terminated import Terminated
from .rate import Rate
from .rates import Rates

if TYPE_CHECKING:
    from ...toplevel.scene import Scene


class BaseAbsoluteRate:
    __slots__ = ()

    def at(self) -> float:
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
        self,
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

    def at(self) -> float:
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


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ScheduleInfo:
    parent_absolute_rate: BaseAbsoluteRate
    rate: Rate
    run_time_scale: float
    run_alpha_scale: float
    launch_condition: Condition
    terminate_condition: Condition


@dataclass(
    kw_only=True,
    slots=True
)
class AnimatingState:
    pass


@dataclass(
    kw_only=True,
    slots=True
)
class BeforeAnimating(AnimatingState):
    launch_condition: Condition


@dataclass(
    kw_only=True,
    slots=True
)
class OnAnimating(AnimatingState):
    timeline_coroutine: Coroutine[None, None, None]
    absolute_rate: AbsoluteRate
    terminate_condition: Condition
    progress_condition: Condition
    children: "list[Animation]"


@dataclass(
    kw_only=True,
    slots=True
)
class AfterAnimating(AnimatingState):
    pass


class Animation(ABC):
    __slots__ = (
        "__weakref__",
        "_run_alpha",
        "_schedule_info",
        "_animating_state"
        #"_animation_state",
        #"_parent_absolute_rate",
        #"_relative_rate",
        #"_launch_condition",
        #"_terminate_condition",
        #"_absolute_rate",
        #"_progress_condition",
        #"_children",
        #"_timeline_coroutine"
    )

    def __init__(
        self,
        # The accumulated alpha value of `timeline`.
        # Left as `inf` if infinite or indefinite.
        # This parameter is required mostly for the program to know
        # how long the animation is before running the timeline.
        run_alpha: float = float("inf")
    ) -> None:
        super().__init__()
        self._run_alpha: float = run_alpha

        # To be initialized when scheduled.
        self._schedule_info: ScheduleInfo | None = None
        self._animating_state: AnimatingState | None = None

        #self._animation_state: AnimationState = AnimationState.UNSCHEDULED
        ## Alive in `AnimationState.BEFORE_ANIMATION`, `AnimationState.ON_ANIMATION`.
        #self._parent_absolute_rate: Rate | None = None
        #self._relative_rate: Rate | None = None
        #self._launch_condition: Condition | None = None
        #self._terminate_condition: Condition | None = None
        ## Alive in `AnimationState.ON_ANIMATION`.
        #self._absolute_rate: Rate | None = None
        #self._progress_condition: Condition | None = None
        #self._children: list[Animation] | None = None
        #self._timeline_coroutine: Coroutine[None, None, None] | None = None

    def _schedule(
        self,
        schedule_info: ScheduleInfo
        #parent_absolute_rate: Rate,
        #relative_rate: Rate,
        #launch_condition: Condition,
        #terminate_condition: Condition
    ) -> None:
        assert self._schedule_info is None
        assert self._animating_state is None
        #schedule_info = ScheduleInfo(
        #    parent_absolute_rate=parent_absolute_rate,
        #    relative_rate=relative_rate,
        #    launch_condition=launch_condition,
        #    terminate_condition=terminate_condition
        #)
        self._schedule_info = schedule_info
        self._animating_state = BeforeAnimating(
            launch_condition=schedule_info.launch_condition
        )
        #assert self._animation_state == AnimationState.UNSCHEDULED
        #self._animation_state = AnimationState.BEFORE_ANIMATION
        #self._parent_absolute_rate = parent_absolute_rate
        #self._relative_rate = relative_rate
        #self._launch_condition = launch_condition
        #self._terminate_condition = terminate_condition

    def _root_schedule(self) -> None:
        self._schedule(ScheduleInfo(
            parent_absolute_rate=BaseAbsoluteRate(),
            rate=Rates.linear(),
            run_time_scale=1.0,
            run_alpha_scale=1.0,
            launch_condition=Conditions.always(),
            terminate_condition=Conditions.never()
        ))

    def _progress(self) -> None:
        #assert (schedule_info := self._schedule_info) is not None
        #assert (animating_state := self._animating_state) is not None
        #animating_state = self._animating_state
        if (animating_state := self.get_before_animating_state()) is not None:
            if not animating_state.launch_condition.judge():
                return
            self.launch()
            #self._animating_state = animating_state = OnAnimating(
            #    timeline_coroutine=self.timeline(),
            #    absolute_rate=AbsoluteRate(
            #        parent_absolute_rate=schedule_info.parent_absolute_rate,
            #        relative_rate=schedule_info.relative_rate
            #    ),
            #    terminate_condition=schedule_info.terminate_condition,
            #    progress_condition=Always(),
            #    children=[]
            #)
        if (animating_state := self.get_on_animating_state()) is not None:
            self._animate_instant(animating_state.absolute_rate.at())
            while not animating_state.terminate_condition.judge():
                for child in animating_state.children[:]:
                    child._progress()
                    if child.get_after_animating_state() is not None:
                        animating_state.children.remove(child)
                #assert (progress_condition := self._progress_condition) is not None
                if not animating_state.progress_condition.judge():
                    return
                try:
                    animating_state.timeline_coroutine.send(None)
                except StopIteration:
                    break
            self.terminate()
            #self._animating_state = animating_state = AfterAnimating()
        if (animating_state := self.get_after_animating_state()) is not None:
            return

    #def _launch(self) -> None:
    #    assert self._animation_state == AnimationState.BEFORE_ANIMATION
    #    assert (parent_absolute_rate := self._parent_absolute_rate) is not None
    #    assert (relative_rate := self._relative_rate) is not None
    #    self._animation_state = AnimationState.ON_ANIMATION
    #    self._timeline_coroutine = self.timeline()
    #    self._absolute_rate = AbsoluteRate(
    #        parent_absolute_rate=parent_absolute_rate,
    #        relative_rate=relative_rate,
    #        timestamp=Toplevel.scene._timestamp
    #    )
    #    self._progress_condition = Always()
    #    self._children = []
    #    self.update(0.0)

    #def _terminate(self) -> None:
    #    assert self._animation_state == AnimationState.ON_ANIMATION
    #    self._animation_state = AnimationState.AFTER_ANIMATION
    #    self._parent_absolute_rate = None
    #    self._relative_rate = None
    #    self._launch_condition = None
    #    self._terminate_condition = None
    #    self._timeline_coroutine = None
    #    self._absolute_rate = None
    #    self._progress_condition = None
    #    self._children = None
    #    if (run_alpha := self._run_alpha) != float("inf"):
    #        self.update(run_alpha)

        #if self._animation_state in (AnimationState.UNSCHEDULED, AnimationState.AFTER_ANIMATION):
        #    raise TypeError
        #assert (launch_condition := self._launch_condition) is not None
        #assert (terminate_condition := self._terminate_condition) is not None
        #if self._animation_state == AnimationState.BEFORE_ANIMATION:
        #    if not launch_condition.judge():
        #        return
        #    self._launch()
        #assert self._animation_state == AnimationState.ON_ANIMATION
        #assert (absolute_rate := self._absolute_rate) is not None
        #assert (children := self._children) is not None
        #assert (timeline_coroutine := self._timeline_coroutine) is not None
        #self.update(absolute_rate.at(Toplevel.scene._timestamp))
        #while not terminate_condition.judge():
        #    for child in children[:]:
        #        child._progress()
        #        if child._animation_state == AnimationState.AFTER_ANIMATION:
        #            children.remove(child)
        #    assert (progress_condition := self._progress_condition) is not None
        #    if not progress_condition.judge():
        #        return
        #    try:
        #        timeline_coroutine.send(None)
        #    except StopIteration:
        #        break
        #self._terminate()

    def _animate_instant(
        self,
        alpha: float
    ) -> None:
        pass

    @abstractmethod
    async def timeline(self) -> None:
        pass

    def get_before_animating_state(self) -> BeforeAnimating | None:
        if isinstance(animating_state := self._animating_state, BeforeAnimating):
            return animating_state
        return None

    def get_on_animating_state(self) -> OnAnimating | None:
        if isinstance(animating_state := self._animating_state, OnAnimating):
            return animating_state
        return None

    def get_after_animating_state(self) -> AfterAnimating | None:
        if isinstance(animating_state := self._animating_state, AfterAnimating):
            return animating_state
        return None

    def launch(self) -> None:
        assert self.get_before_animating_state() is not None
        assert (schedule_info := self._schedule_info) is not None
        self._animating_state = OnAnimating(
            timeline_coroutine=self.timeline(),
            absolute_rate=AbsoluteRate(
                parent_absolute_rate=schedule_info.parent_absolute_rate,
                rate=schedule_info.rate,
                run_time_scale=schedule_info.run_alpha_scale,
                run_alpha_scale=schedule_info.run_alpha_scale
                #relative_rate=schedule_info.relative_rate
            ),
            terminate_condition=schedule_info.terminate_condition,
            progress_condition=Conditions.always(),
            children=[]
        )

    def terminate(self) -> None:
        assert self.get_on_animating_state() is not None
        assert self._schedule_info is not None
        self._animating_state = AfterAnimating()

    def prepare(
        self,
        animation: "Animation",
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
        #assert self._animation_state == AnimationState.ON_ANIMATION
        #assert (absolute_rate := self._absolute_rate) is not None
        assert isinstance(animating_state := self._animating_state, OnAnimating)
        if (run_alpha := animation._run_alpha) == float("inf"):
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
        assert rate.is_increasing()
        animation._schedule(ScheduleInfo(
            parent_absolute_rate=animating_state.absolute_rate,
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
        animating_state.children.append(animation)

    async def wait_until(
        self,
        progress_condition: Condition
    ) -> None:
        #assert self._animation_state == AnimationState.ON_ANIMATION
        #assert self._progress_condition is not None
        assert isinstance(animating_state := self._animating_state, OnAnimating)
        animating_state.progress_condition = progress_condition
        await asyncio.sleep(0.0)

    # shortcuts

    @property
    def scene(self) -> "Scene":
        return Toplevel.scene

    async def play(
        self,
        animation: "Animation",
        rate: Rate | None = None,
        run_time: float | None = None,
        launch_condition: Condition = Conditions.always(),
        terminate_condition: Condition = Conditions.never()
    ) -> None:
        self.prepare(
            animation,
            rate=rate,
            run_time=run_time,
            launch_condition=launch_condition,
            terminate_condition=terminate_condition
        )
        await self.wait_until(animation.terminated())

    async def wait(
        self,
        delta_alpha: float = 1.0
    ) -> None:
        await self.wait_until(self.progressed_duration(delta_alpha))

    # conditions

    def launched(self) -> "AnimationLaunchedCondition":
        return AnimationLaunchedCondition(self)

    def terminated(self) -> "AnimationTerminatedCondition":
        return AnimationTerminatedCondition(self)

    def progressed_duration(
        self,
        delta_alpha: float
    ) -> "AnimationProgressedDurationCondition":
        return AnimationProgressedDurationCondition(self, delta_alpha)

    #async def wait_forever(self) -> None:
    #    await self.wait_until(Never())


class AnimationLaunchedCondition(Condition):
    __slots__ = ("_animation_ref",)

    def __init__(
        self,
        animation: Animation
    ) -> None:
        super().__init__()
        self._animation_ref: weakref.ref[Animation] = weakref.ref(animation)

    def judge(self) -> bool:
        animation = self._animation_ref()
        return (
            animation is None
            or animation.get_on_animating_state() is not None
            or animation.get_after_animating_state() is not None
        )


class AnimationTerminatedCondition(Condition):
    __slots__ = ("_animation_ref",)

    def __init__(
        self,
        animation: Animation
    ) -> None:
        super().__init__()
        self._animation_ref: weakref.ref[Animation] = weakref.ref(animation)

    def judge(self) -> bool:
        animation = self._animation_ref()
        return (
            animation is None
            or animation.get_after_animating_state() is not None
        )


class AnimationProgressedDurationCondition(Condition):
    __slots__ = (
        "_animation_ref",
        "_target_alpha"
    )

    def __init__(
        self,
        animation: Animation,
        delta_alpha: float
    ) -> None:
        assert (animating_state := animation.get_on_animating_state()) is not None
        self._animation_ref: weakref.ref[Animation] = weakref.ref(animation)
        self._target_alpha: float = animating_state.absolute_rate.at() + delta_alpha

    def judge(self) -> bool:
        animation = self._animation_ref()
        return (
            animation is None
            or (animating_state := animation.get_on_animating_state()) is None
            or animating_state.absolute_rate.at() >= self._target_alpha
        )
