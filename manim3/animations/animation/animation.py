import asyncio
from abc import (
    ABC,
    abstractmethod
)
from typing import (
    TYPE_CHECKING,
    Coroutine
)

from ...toplevel.toplevel import Toplevel
from .animation_state import AnimationState
from .conditions.always import Always
from .conditions.condition import Condition
from .conditions.never import Never
from .conditions.progress_duration import ProgressDuration
from .conditions.terminated import Terminated
from .rates.linear import Linear
from .rates.rate import Rate

if TYPE_CHECKING:
    from ...toplevel.scene import Scene


class AbsoluteRate(Rate):
    __slots__ = (
        "_parent_absolute_rate",
        "_relative_rate",
        "_initial_alpha"
    )

    def __init__(
        self,
        parent_absolute_rate: Rate,
        relative_rate: Rate,
        timestamp: float
    ) -> None:
        super().__init__()
        self._parent_absolute_rate: Rate = parent_absolute_rate
        self._relative_rate: Rate = relative_rate
        self._initial_alpha: float = parent_absolute_rate.at(timestamp)

    def at(
        self,
        t: float
    ) -> float:
        return self._relative_rate.at(self._parent_absolute_rate.at(t) - self._initial_alpha)


class ScaledRate(Rate):
    __slots__ = (
        "_rate",
        "_run_time_scale",
        "_run_alpha_scale"
    )

    def __init__(
        self,
        rate: Rate,
        run_time_scale: float,
        run_alpha_scale: float
    ) -> None:
        super().__init__()
        self._rate: Rate = rate
        self._run_time_scale: float = run_time_scale
        self._run_alpha_scale: float = run_alpha_scale

    def at(
        self,
        t: float
    ) -> float:
        return self._rate.at(t / self._run_time_scale) * self._run_alpha_scale


class Animation(ABC):
    __slots__ = (
        "__weakref__",
        "_run_alpha",
        "_animation_state",
        "_parent_absolute_rate",
        "_relative_rate",
        "_launch_condition",
        "_terminate_condition",
        "_absolute_rate",
        "_progress_condition",
        "_children",
        "_timeline_coroutine"
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

        self._animation_state: AnimationState = AnimationState.UNBOUND
        # Alive in `AnimationState.BEFORE_ANIMATION`, `AnimationState.ON_ANIMATION`.
        self._parent_absolute_rate: Rate | None = None
        self._relative_rate: Rate | None = None
        self._launch_condition: Condition | None = None
        self._terminate_condition: Condition | None = None
        # Alive in `AnimationState.ON_ANIMATION`.
        self._absolute_rate: Rate | None = None
        self._progress_condition: Condition | None = None
        self._children: list[Animation] | None = None
        self._timeline_coroutine: Coroutine[None, None, None] | None = None

    def _schedule(
        self,
        # `[0.0, +infty) -> [0.0, +infty), time |-> alpha`
        # Must be an increasing function.
        relative_rate: Rate = Linear(),
        parent_absolute_rate: Rate = Linear(),
        launch_condition: Condition = Always(),
        terminate_condition: Condition = Never()
    ) -> None:
        assert self._animation_state == AnimationState.UNBOUND
        self._animation_state = AnimationState.BEFORE_ANIMATION
        self._parent_absolute_rate = parent_absolute_rate
        self._relative_rate = relative_rate
        self._launch_condition = launch_condition
        self._terminate_condition = terminate_condition

    def _launch(self) -> None:
        assert self._animation_state == AnimationState.BEFORE_ANIMATION
        assert (parent_absolute_rate := self._parent_absolute_rate) is not None
        assert (relative_rate := self._relative_rate) is not None
        self._animation_state = AnimationState.ON_ANIMATION
        self._timeline_coroutine = self.timeline()
        self._absolute_rate = AbsoluteRate(
            parent_absolute_rate=parent_absolute_rate,
            relative_rate=relative_rate,
            timestamp=Toplevel.scene._timestamp
        )
        self._progress_condition = Always()
        self._children = []
        self.updater(0.0)

    def _terminate(self) -> None:
        assert self._animation_state == AnimationState.ON_ANIMATION
        self._animation_state = AnimationState.AFTER_ANIMATION
        self._parent_absolute_rate = None
        self._relative_rate = None
        self._launch_condition = None
        self._terminate_condition = None
        self._timeline_coroutine = None
        self._absolute_rate = None
        self._progress_condition = None
        self._children = None
        if (run_alpha := self._run_alpha) != float("inf"):
            self.updater(run_alpha)

    def _progress(self) -> None:
        if self._animation_state in (AnimationState.UNBOUND, AnimationState.AFTER_ANIMATION):
            raise TypeError
        assert (launch_condition := self._launch_condition) is not None
        assert (terminate_condition := self._terminate_condition) is not None
        if self._animation_state == AnimationState.BEFORE_ANIMATION:
            if not launch_condition.judge():
                return
            self._launch()
        assert self._animation_state == AnimationState.ON_ANIMATION
        assert (absolute_rate := self._absolute_rate) is not None
        assert (children := self._children) is not None
        assert (timeline_coroutine := self._timeline_coroutine) is not None
        self.updater(absolute_rate.at(Toplevel.scene._timestamp))
        while not terminate_condition.judge():
            for child in children[:]:
                child._progress()
                if child._animation_state == AnimationState.AFTER_ANIMATION:
                    children.remove(child)
            assert (progress_condition := self._progress_condition) is not None
            if not progress_condition.judge():
                return
            try:
                timeline_coroutine.send(None)
            except StopIteration:
                break
        self._terminate()

    def updater(
        self,
        alpha: float
    ) -> None:
        pass

    @abstractmethod
    async def timeline(self) -> None:
        pass

    def prepare(
        self,
        animation: "Animation",
        *,
        # Forced to be `None` if `_run_alpha` is infinity.
        rate: Rate | None = None,
        # Intepreted as "the inverse of run speed" if `_run_alpha` is infinity.
        run_time: float | None = None,
        launch_condition: Condition = Always(),
        terminate_condition: Condition = Never()
    ) -> None:
        assert self._animation_state == AnimationState.ON_ANIMATION
        assert (absolute_rate := self._absolute_rate) is not None

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
            rate = Linear()
        animation._schedule(
            parent_absolute_rate=absolute_rate,
            relative_rate=ScaledRate(
                rate=rate,
                run_time_scale=run_time_scale,
                run_alpha_scale=run_alpha_scale
            ),
            launch_condition=launch_condition,
            terminate_condition=terminate_condition
        )
        assert (children := self._children) is not None
        children.append(animation)

    async def wait_until(
        self,
        progress_condition: Condition
    ) -> None:
        assert self._animation_state == AnimationState.ON_ANIMATION
        assert self._progress_condition is not None
        self._progress_condition = progress_condition
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
        launch_condition: Condition = Always(),
        terminate_condition: Condition = Never()
    ) -> None:
        self.prepare(
            animation,
            rate=rate,
            run_time=run_time,
            launch_condition=launch_condition,
            terminate_condition=terminate_condition
        )
        await self.wait_until(Terminated(animation))

    async def wait(
        self,
        delta_alpha: float = 1.0
    ) -> None:
        await self.wait_until(ProgressDuration(self, delta_alpha))

    async def wait_forever(self) -> None:
        await self.wait_until(Never())
