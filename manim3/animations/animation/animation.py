from abc import (
    ABC,
    abstractmethod
)
import asyncio
from typing import (
    TYPE_CHECKING,
    Callable,
    Coroutine
)

from ...toplevel.toplevel import Toplevel
from .animation_state import AnimationState
from .conditions.always import Always
from .conditions.condition import Condition
from .conditions.never import Never
from .conditions.progress_duration import ProgressDuration
from .conditions.terminated import Terminated
from .rates import Rates

if TYPE_CHECKING:
    from ...toplevel.scene import Scene


class Animation(ABC):
    __slots__ = (
        "__weakref__",
        #"_updater",
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
        #self._updater: Callable[[float], None] | None = updater
        self._run_alpha: float = run_alpha

        self._animation_state: AnimationState = AnimationState.UNBOUND
        # Alive in `AnimationState.BEFORE_ANIMATION`, `AnimationState.ON_ANIMATION`.
        self._parent_absolute_rate: Callable[[float], float] | None = None
        self._relative_rate: Callable[[float], float] | None = None
        self._launch_condition: Condition | None = None
        self._terminate_condition: Condition | None = None
        # Alive in `AnimationState.ON_ANIMATION`.
        self._absolute_rate: Callable[[float], float] | None = None
        self._progress_condition: Condition | None = None
        self._children: list[Animation] | None = None
        self._timeline_coroutine: Coroutine[None, None, None] | None = None

    def _schedule(
        self,
        # `[0.0, +infty) -> [0.0, +infty), time |-> alpha`
        # Must be an increasing function.
        relative_rate: Callable[[float], float] = Rates.linear,
        parent_absolute_rate: Callable[[float], float] = Rates.linear,
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

        def make_absolute_rate(
            parent_absolute_rate: Callable[[float], float],
            relative_rate: Callable[[float], float],
            timestamp: float
        ) -> Callable[[float], float]:
            #print([
            #    (i/4, parent_absolute_rate(i/4))
            #    for i in range(11)
            #])
            #print([
            #    (i/4, relative_rate(i/4))
            #    for i in range(11)
            #])
            #print(timestamp)

            def result(
                t: float
            ) -> float:
                return relative_rate(parent_absolute_rate(t) - parent_absolute_rate(timestamp))

            #print([
            #    (i/4, result(i/4))
            #    for i in range(11)
            #])
            #print()
            return result

        assert self._animation_state == AnimationState.BEFORE_ANIMATION
        assert (parent_absolute_rate := self._parent_absolute_rate) is not None
        assert (relative_rate := self._relative_rate) is not None
        self._animation_state = AnimationState.ON_ANIMATION
        self._timeline_coroutine = self.timeline()
        self._absolute_rate = make_absolute_rate(
            parent_absolute_rate=parent_absolute_rate,
            relative_rate=relative_rate,
            timestamp=Toplevel.scene._timestamp
        )
        #self._absolute_rate = RateUtils.compose_rates(
        #    RateUtils.lag_rate(
        #        relative_rate,
        #        lag_time=parent_absolute_rate(Toplevel.scene._timestamp)
        #    ),
        #    parent_absolute_rate
        #)
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
            if not launch_condition._judge():
                return
            self._launch()
        assert self._animation_state == AnimationState.ON_ANIMATION
        assert (absolute_rate := self._absolute_rate) is not None
        assert (children := self._children) is not None
        assert (timeline_coroutine := self._timeline_coroutine) is not None
        self.updater(absolute_rate(Toplevel.scene._timestamp))
        while not terminate_condition._judge():
            for child in children[:]:
                child._progress()
                if child._animation_state == AnimationState.AFTER_ANIMATION:
                    children.remove(child)
            assert (progress_condition := self._progress_condition) is not None
            if not progress_condition._judge():
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

    def prepare(
        self,
        animation: "Animation",
        *,
        # If provided, should be defined on `[0, 1] -> [0, 1]` and increasing.
        # Forced to be `None` if `_run_alpha` is infinity.
        rate: Callable[[float], float] | None = None,
        # Intepreted as "the inverse of run speed" if `_run_alpha` is infinity.
        run_time: float = 1.0,
        launch_condition: Condition = Always(),
        terminate_condition: Condition = Never()
    ) -> None:

        def scale_rate(
            rate: Callable[[float], float],
            run_time_scale: float,
            run_alpha_scale: float
        ) -> Callable[[float], float]:

            def result(
                t: float
            ) -> float:
                return rate(t / run_time_scale) * run_alpha_scale

            return result

        assert self._animation_state == AnimationState.ON_ANIMATION
        assert (absolute_rate := self._absolute_rate) is not None
        assert (run_alpha := animation._run_alpha) != float("inf") or rate is None
        relative_rate = scale_rate(
            rate=rate if rate is not None else Rates.linear,
            run_time_scale=run_time,
            run_alpha_scale=run_alpha if run_alpha != float("inf") else 1.0
        )
        animation._schedule(
            parent_absolute_rate=absolute_rate,
            relative_rate=relative_rate,
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

    @abstractmethod
    async def timeline(self) -> None:
        pass

    # shortcuts

    @property
    def scene(self) -> "Scene":
        return Toplevel.scene

    async def play(
        self,
        animation: "Animation",
        rate: Callable[[float], float] | None = None,
        run_time: float = 1.0
    ) -> None:
        self.prepare(animation, rate=rate, run_time=run_time)
        await self.wait_until(Terminated(animation))

    async def wait(
        self,
        delta_alpha: float = 1.0
    ) -> None:
        await self.wait_until(ProgressDuration(self, delta_alpha))

    async def wait_forever(self) -> None:
        await self.wait_until(Never())
