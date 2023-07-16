from abc import (
    ABC,
    abstractmethod
)
import asyncio
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Coroutine,
    Iterable
)

from ..toplevel.toplevel import Toplevel
from ..utils.rate import RateUtils

if TYPE_CHECKING:
    from ..toplevel.scene import Scene


#class Toplevel:
#    __slots__ = ()

#    _timestamp: float
#    _toplevel_animation: "Animation" = NotImplemented
#    _active_animation_to_scene_dict: "weakref.WeakValueDictionary[Animation, Scene]" = weakref.WeakValueDictionary()

#    def __new__(cls):
#        raise TypeError

#    @classmethod
#    def bind_animation_to_scene(
#        cls,
#        animation: "Animation",
#        scene: "Scene"
#    ) -> None:
#        assert animation not in cls._active_animation_to_scene_dict
#        cls._active_animation_to_scene_dict[animation] = scene

#    @classmethod
#    def get_scene_by_animation(
#        cls,
#        animation: "Animation"
#    ) -> "Scene":
#        return cls._active_animation_to_scene_dict[animation]

#    @classmethod
#    def remove_animation(
#        cls,
#        animation: "Animation"
#    ) -> None:
#        cls._active_animation_to_scene_dict.pop(animation)

#    @classmethod
#    @contextmanager
#    def set_animation(
#        cls,
#        animation: "Animation"
#    ) -> Iterator[None]:
#        stored_toplevel_animation = cls._toplevel_animation
#        cls._toplevel_animation = animation
#        yield
#        cls._toplevel_animation = stored_toplevel_animation

#    @classmethod
#    def get_scene(cls) -> "Scene":
#        return cls._active_animation_to_scene_dict[cls._toplevel_animation]


class AnimationState(Enum):
    UNBOUND = 0
    BEFORE_ANIMATION = 1
    ON_ANIMATION = 2
    AFTER_ANIMATION = 3


class Animation(ABC):
    __slots__ = (
        "_updater",
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
        # Update continuously (per frame).
        updater: Callable[[float], None] | None = None
        #launch_condition: Callable[[], bool] | None = None,
        #terminate_condition: Callable[[], bool] | None = None
    ) -> None:
        super().__init__()
        self._updater: Callable[[float], None] | None = updater

        self._animation_state: AnimationState = AnimationState.UNBOUND
        # Alive in `AnimationState.BEFORE_ANIMATION`, `AnimationState.ON_ANIMATION`.
        self._parent_absolute_rate: Callable[[float], float] | None = None
        self._relative_rate: Callable[[float], float] | None = None
        self._launch_condition: Callable[[], bool] | None = None
        self._terminate_condition: Callable[[], bool] | None = None
        # Alive in `AnimationState.ON_ANIMATION`.
        self._absolute_rate: Callable[[float], float] | None = None
        self._progress_condition: Callable[[], bool] | None = None
        self._children: list[Animation] | None = None
        self._timeline_coroutine: Coroutine[None, None, None] | None = None

    def _prepare_animation(
        self,
        # `[0.0, +infty) -> [0.0, +infty), time |-> alpha`
        # Must be an increasing function.
        relative_rate: Callable[[float], float] = RateUtils.linear,
        parent_absolute_rate: Callable[[float], float] = RateUtils.linear,
        launch_condition: Callable[[], bool] | None = None,
        terminate_condition: Callable[[], bool] | None = None
    ) -> None:
        assert self._animation_state == AnimationState.UNBOUND
        if launch_condition is None:
            launch_condition = self.always()
        if terminate_condition is None:
            terminate_condition = self.never()
        self._animation_state = AnimationState.BEFORE_ANIMATION
        self._parent_absolute_rate = parent_absolute_rate
        self._relative_rate = relative_rate
        self._launch_condition = launch_condition
        self._terminate_condition = terminate_condition

    def _launch_animation(self) -> None:
        assert self._animation_state == AnimationState.BEFORE_ANIMATION
        assert (parent_absolute_rate := self._parent_absolute_rate) is not None
        assert (relative_rate := self._relative_rate) is not None
        self._animation_state = AnimationState.ON_ANIMATION
        self._timeline_coroutine = self.timeline()
        self._absolute_rate = RateUtils.adjust(
            RateUtils.compose(
                relative_rate,
                parent_absolute_rate
            ),
            lag_time=Toplevel.scene._timestamp
        )
        self._progress_condition = self.always()
        self._children = []

    def _terminate_animation(self) -> None:
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

    def _progress_animation(self) -> None:
        ##with Toplevel.set_animation(self):
        #while self._progress_condition():
        #    try:
        #        self._timeline_coroutine.send(None)
        #    except StopIteration:
        #        pass
        #        #Toplevel.remove_animation(self)

        if self._animation_state in (AnimationState.UNBOUND, AnimationState.AFTER_ANIMATION):
            raise TypeError
        assert (launch_condition := self._launch_condition) is not None
        assert (terminate_condition := self._terminate_condition) is not None
        if self._animation_state == AnimationState.BEFORE_ANIMATION and launch_condition():
            self._launch_animation()
        assert (timeline_coroutine := self._timeline_coroutine) is not None
        assert (children := self._children) is not None
        while self._animation_state == AnimationState.ON_ANIMATION:
            if terminate_condition():
                self._terminate_animation()
                break
            for child in children[:]:
                child._progress_animation()
                if child._animation_state == AnimationState.AFTER_ANIMATION:
                    children.remove(child)
            assert (progress_condition := self._progress_condition) is not None
            if not progress_condition():
                break
            try:
                timeline_coroutine.send(None)
            except StopIteration:
                self._terminate_animation()
                break

    def _update(
        self,
        timestamp: float
    ) -> None:
        if self._animation_state != AnimationState.ON_ANIMATION:
            return
        assert (children := self._children) is not None
        for child in children:
            child._update(timestamp)
        assert (absolute_rate := self._absolute_rate) is not None
        if (updater := self._updater) is not None:
            updater(absolute_rate(timestamp))

    #def _get_bound_scene(self) -> "Scene":
    #    return Toplevel.get_scene_by_animation(self)

    def prepare(
        self,
        animation: "Animation",
        *,
        relative_rate: Callable[[float], float] = RateUtils.linear,
        launch_condition: Callable[[], bool] | None = None,
        terminate_condition: Callable[[], bool] | None = None
    ) -> None:
        assert self._animation_state == AnimationState.ON_ANIMATION
        assert (absolute_rate := self._absolute_rate) is not None
        animation._prepare_animation(
            parent_absolute_rate=absolute_rate,
            relative_rate=relative_rate,
            launch_condition=launch_condition,
            terminate_condition=terminate_condition
        )
        assert (children := self._children) is not None
        children.append(animation)
        #bound_scene = self._get_bound_scene()
        #for animation in animations:
        #    assert animation._absolute_rate is NotImplemented
        #    animation._absolute_rate = RateUtils.compose(
        #        animation._relative_rate,
        #        self._absolute_rate
        #    )
        #    #Toplevel.bind_animation_to_scene(animation, bound_scene)
        #    self._children.append(animation)
        #    animation._progress_timeline()

    async def set_progress_condition(
        self,
        progress_condition: Callable[[], bool]
    ) -> None:
        assert self._animation_state == AnimationState.ON_ANIMATION
        assert self._progress_condition is not None
        self._progress_condition = progress_condition
        await asyncio.sleep(0.0)

    @abstractmethod
    async def timeline(self) -> None:
        pass

    # conditions

    @classmethod
    def all(
        cls,
        funcs: Iterable[Callable[[], bool]]
    ) -> Callable[[], bool]:

        def result() -> bool:
            return all(func() for func in funcs)

        return result

    @classmethod
    def any(
        cls,
        funcs: Iterable[Callable[[], bool]]
    ) -> Callable[[], bool]:

        def result() -> bool:
            return any(func() for func in funcs)

        return result

    def always(self) -> Callable[[], bool]:

        def result() -> bool:
            return True

        return result

    def never(self) -> Callable[[], bool]:

        def result() -> bool:
            return False

        return result

    def launched(self) -> Callable[[], bool]:

        def result() -> bool:
            return self._animation_state in (AnimationState.ON_ANIMATION, AnimationState.AFTER_ANIMATION)

        return result

    def terminated(self) -> Callable[[], bool]:

        def result() -> bool:
            return self._animation_state == AnimationState.AFTER_ANIMATION

        return result

    def wait_duration(
        self,
        delta_alpha: float
    ) -> Callable[[], bool]:
        assert self._animation_state == AnimationState.ON_ANIMATION
        assert (absolute_rate := self._absolute_rate) is not None
        target_alpha = absolute_rate(Toplevel.scene._timestamp) + delta_alpha

        def result() -> bool:
            return absolute_rate(Toplevel.scene._timestamp) >= target_alpha

        return result

    # shortcuts

    @property
    def scene(self) -> "Scene":
        return Toplevel.scene

    async def play(
        self,
        animation: "Animation",
        rate_func: Callable[[float], float] = RateUtils.linear,
        # This should be treated as "the inverse of run speed",
        # but is equivalent to run time if the run alpha is 1.
        run_time: float | None = None
    ) -> None:
        relative_rate = rate_func if run_time is None else RateUtils.adjust(
            rate_func,
            run_time_scale=run_time
        )
        self.prepare(animation, relative_rate=relative_rate)
        await self.set_progress_condition(animation.terminated())

    async def wait(
        self,
        delta_alpha: float = 1.0
    ) -> None:
        #target_alpha = self._absolute_rate(Toplevel._timestamp) + delta_alpha
        #await self.set_progress_condition(lambda: self._absolute_rate(Toplevel._timestamp) >= target_alpha)
        await self.set_progress_condition(self.wait_duration(delta_alpha))

    async def wait_forever(self) -> None:
        await self.set_progress_condition(self.never())
