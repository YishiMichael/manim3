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


class ProgressCondition(ABC):
    __slots__ = ()

    @abstractmethod
    def evaluate(
        self,
        animation: "Animation"
    ) -> bool:
        pass


class AlwaysTrue(ProgressCondition):
    __slots__ = ()

    def evaluate(
        self,
        animation: "Animation"
    ) -> bool:
        return True


class AlwaysFalse(ProgressCondition):
    __slots__ = ()

    def evaluate(
        self,
        animation: "Animation"
    ) -> bool:
        return False


class Animation(ABC):
    __slots__ = (
        "_updater",
        "_relative_rate",
        "_absolute_rate",
        "_children",
        "_progress_condition",
        "_timeline_coroutine"
    )

    def __init__(
        self,
        # Update continuously (per frame).
        updater: Callable[[float], None] | None = None,
        # `[0.0, +infty) -> [0.0, +infty), time |-> alpha`
        # Must be an increasing function.
        relative_rate: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__()
        self._updater: Callable[[float], None] | None = updater
        self._relative_rate: Callable[[float], float] = relative_rate
        self._absolute_rate: Callable[[float], float] = NotImplemented
        self._children: list[Animation] = []
        self._progress_condition: Callable[[], bool] = lambda: True
        self._timeline_coroutine: Coroutine[None, None, None] = self.timeline()

    def _progress_timeline(self) -> None:
        #with Toplevel.set_animation(self):
        while self._progress_condition():
            try:
                self._timeline_coroutine.send(None)
            except StopIteration:
                pass
                #Toplevel.remove_animation(self)

    #def _get_bound_scene(self) -> "Scene":
    #    return Toplevel.get_scene_by_animation(self)

    def prepare(
        self,
        *animations: "Animation"
    ) -> None:
        #bound_scene = self._get_bound_scene()
        for animation in animations:
            assert animation._absolute_rate is NotImplemented
            animation._absolute_rate = RateUtils.compose(
                animation._relative_rate,
                self._absolute_rate
            )
            #Toplevel.bind_animation_to_scene(animation, bound_scene)
            self._children.append(animation)
            animation._progress_timeline()

    async def set_continue_condition(
        self,
        continue_condition: Callable[[], bool]
    ) -> None:
        self._continue_condition = continue_condition
        await asyncio.sleep(0.0)

    async def play(
        self,
        animation: "Animation"
    ) -> None:
        self.prepare(animation)
        await self.set_continue_condition(lambda: animation not in Toplevel._active_animation_to_scene_dict)

    async def wait(
        self,
        delta_alpha: float = 1.0
    ) -> None:
        target_alpha = self._absolute_rate(Toplevel._timestamp) + delta_alpha
        await self.set_continue_condition(lambda: self._absolute_rate(Toplevel._timestamp) >= target_alpha)

    async def wait_forever(self) -> None:
        await self.set_continue_condition(lambda: False)

    @abstractmethod
    async def timeline(self) -> None:
        pass

    # Access the scene the animation is operated on.
    # Always accessible in the body of `timeline()` method.
    @property
    def scene(self) -> "Scene":
        return Toplevel.scene
