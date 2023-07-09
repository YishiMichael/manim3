from abc import (
    ABC,
    abstractmethod
)
import asyncio
from dataclasses import dataclass
import itertools as it
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Iterator
)
import weakref

from ..utils.rate import RateUtils

if TYPE_CHECKING:
    from ..scene.scene import Scene


#class TimelineState(Enum):
#    START = 1
#    STOP = -1
#    AWAIT = 0


class Toplevel:
    __slots__ = ()

    _animation_id_to_scene_dict: "weakref.WeakValueDictionary[int, Scene]" = weakref.WeakValueDictionary()
    _toplevel_animation_id: int

    def __new__(cls):
        raise TypeError

    @classmethod
    def bind_animation_id_to_scene(
        cls,
        animation_id: int,
        scene: "Scene"
    ) -> None:
        assert animation_id not in cls._animation_id_to_scene_dict
        cls._animation_id_to_scene_dict[animation_id] = scene

    @classmethod
    def get_scene_by_animation_id(
        cls,
        animation_id: int
    ) -> "Scene":
        return cls._animation_id_to_scene_dict[animation_id]

    @classmethod
    def set_animation_id(
        cls,
        animation_id: int
    ) -> None:
        cls._toplevel_animation_id = animation_id

    @classmethod
    def get_scene(cls) -> "Scene":
        return cls._animation_id_to_scene_dict[cls._toplevel_animation_id]


@dataclass(
    kw_only=True,
    slots=True
)
class TimelineSignal:
    timestamp: float
    #animation: "Animation"
    #absolute_rate: Callable[[float], float] | None
    #timeline_state: TimelineState


@dataclass(
    kw_only=True,
    slots=True
)
class TimelineStartSignal(TimelineSignal):
    animation_id: int
    composed_updater: Callable[[float], None] | None


@dataclass(
    kw_only=True,
    slots=True
)
class TimelineStopSignal(TimelineSignal):
    animation_id: int


@dataclass(
    kw_only=True,
    slots=True
)
class TimelineAwaitSignal(TimelineSignal):
    pass


@dataclass(
    kw_only=True,
    slots=True
)
class TimelineProgress:
    timeline_start_alpha: float
    signal: TimelineSignal
    signal_alpha: float


class TimelineManager:
    __slots__ = ("timelines",)

    def __init__(self) -> None:
        super().__init__()
        self.timelines: dict[Iterator[TimelineSignal], TimelineProgress] = {}
        #self.start_alpha_dict: dict[Iterator[TimelineSignal], float] = {}
        #self.signal_dict: dict[Iterator[TimelineSignal], TimelineSignal] = {}

    def add_timeline_at(
        self,
        timeline: Iterator[TimelineSignal],
        timeline_start_alpha: float
    ) -> None:
        signal = next(timeline)
        self.timelines[timeline] = TimelineProgress(
            timeline_start_alpha=timeline_start_alpha,
            signal=signal,
            signal_alpha=timeline_start_alpha + signal.timestamp
        )
        #self.start_alpha_dict[timeline] = start_alpha
        #self.signal_dict[timeline] = signal

    def advance_to_next_signal(
        self,
        timeline: Iterator[TimelineSignal]
    ) -> None:
        try:
            signal = next(timeline)
        except StopIteration:
            self.timelines.pop(timeline)
            return
        progress = self.timelines[timeline]
        progress.signal = signal
        progress.signal_alpha = progress.timeline_start_alpha + signal.timestamp

    def iter_timeline_progress_until(
        self,
        terminate_alpha: float# | None
    ) -> Iterator[TimelineProgress]:
        timelines = self.timelines

        def get_rounded_signal_alpha(
            timeline: Iterator[TimelineSignal]
        ) -> float:
            return round(timelines[timeline].signal_alpha, 3)  # Avoid floating error.

        rounded_terminate_alpha = round(terminate_alpha, 3)# if terminate_alpha is not None else None
        while timelines:
            timeline = min(timelines, key=get_rounded_signal_alpha)
            rounded_signal_alpha = get_rounded_signal_alpha(timeline)
            if rounded_terminate_alpha is not None and rounded_signal_alpha > rounded_terminate_alpha:
                return

            yield timelines[timeline]
            self.advance_to_next_signal(timeline)






        #start_alpha_dict = self.start_alpha_dict
        #signal_dict = self.signal_dict

        #def get_signal_item(
        #    located_signal_timeline: LocatedSignalTimeline
        #) -> tuple[TimelineSignal, float, float]:
        #    signal = located_signal_timeline.signal
        #    signal_timeline_start_alpha = located_signal_timeline.start_alpha
        #    signal_alpha = signal_timeline_start_alpha + signal.timestamp
        #    return signal, signal_timeline_start_alpha, signal_alpha
        #    #next_alpha = start_alpha_dict[signal_timeline] + signal_dict[signal_timeline].timestamp
        #    #return round(next_alpha, 3)  # Avoid floating error.

        #rounded_terminate_alpha = round(terminate_alpha, 3) if terminate_alpha is not None else None
        #for signal_item in sorted((
        #    get_signal_item(located_signal_timeline)
        #    for located_signal_timeline in self.timelines
        #), key=lambda t: round(t[2], 3)):

    #def is_not_empty(self) -> bool:
    #    return bool(self.signal_dict)

    #def get_next_signal_timeline_item(self) -> tuple[Iterator[TimelineSignal], float, TimelineSignal]:
    #    start_alpha_dict = self.start_alpha_dict
    #    signal_dict = self.signal_dict

    #    def get_next_alpha(
    #        signal_timeline: Iterator[TimelineSignal]
    #    ) -> float:
    #        next_alpha = start_alpha_dict[signal_timeline] + signal_dict[signal_timeline].timestamp
    #        return round(next_alpha, 3)  # Avoid floating error.

    #    signal_timeline = min(signal_dict, key=get_next_alpha)
    #    return signal_timeline, start_alpha_dict[signal_timeline], signal_dict[signal_timeline]


class Animation(ABC):
    __slots__ = (
        "_id",
        "_updater",
        "_run_time",
        "_relative_rate",
        "_new_children",
        "_delta_alpha"
        #"_is_prepared_flag",
        #"_scene_ref"
    )

    _id_counter: ClassVar[it.count] = it.count()

    def __init__(
        self,
        # Update continuously (per frame).
        updater: Callable[[float], None] | None = None,
        # If provided, the animation will be clipped from right at `run_time`.
        # `parent.play(self)` will call `parent.wait(run_time)` that covers this animation.
        run_time: float | None = None,
        # `[0.0, run_time] -> [0.0, +infty), time |-> alpha`
        # Must be an increasing function.
        relative_rate: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__()
        self._id: int = next(type(self)._id_counter)
        self._updater: Callable[[float], None] | None = updater
        self._run_time: float | None = run_time
        self._relative_rate: Callable[[float], float] = relative_rate
        self._new_children: list[Animation] = []
        self._delta_alpha: float = 0.0
        # The structure of animations forms a tree without any reoccurances of nodes.
        # This marks whether the node already exists in the tree.
        #self._is_prepared_flag: bool = False
        #self._scene_ref: "weakref.ref[Scene] | None" = None

    def _wait_timeline(self) -> "Iterator[tuple[list[Animation], float]]":
        timeline_coroutine = self.timeline()
        try:
            while True:
                timeline_coroutine.send(None)
                yield self._new_children, self._delta_alpha
                self._new_children.clear()
        except StopIteration:
            pass

    def _signal_timeline(self) -> Iterator[TimelineSignal]:

        def compose_updater(
            updater: Callable[[float], None] | None,
            relative_rate: Callable[[float], float]
        ) -> Callable[[float], None] | None:
            if updater is None:
                return None

            def result(
                t: float
            ) -> None:
                return updater(relative_rate(t))

            return result

        animation_id = self._id
        relative_rate = self._relative_rate
        relative_rate_inv = RateUtils.inverse(relative_rate)

        run_time = self._run_time
        assert run_time is None or run_time >= 0.0
        run_alpha = relative_rate(run_time) if run_time is not None else None
        current_time = 0.0
        current_alpha = relative_rate(current_time)
        assert current_alpha >= 0.0

        manager = TimelineManager()

        #yield TimelineSignal(
        #    timestamp=0.0,
        #    animation=self,
        #    absolute_rate=relative_rate,
        #    timeline_state=TimelineState.START
        #)
        yield TimelineStartSignal(
            timestamp=current_time,
            animation_id=animation_id,
            composed_updater=compose_updater(self._updater, relative_rate)
            #absolute_rate=relative_rate
        )

        for new_children, wait_delta_alpha in self._wait_timeline():
            for child in new_children:
                manager.add_timeline_at(
                    timeline=child._signal_timeline(),
                    timeline_start_alpha=current_alpha
                )
            #self._new_children.clear()

            assert wait_delta_alpha >= 0.0
            current_alpha += wait_delta_alpha
            if run_alpha is not None and current_alpha > run_alpha:
                early_break = True
                current_alpha = run_alpha
            else:
                early_break = False
            current_time = relative_rate_inv(current_alpha)

            for progress in manager.iter_timeline_progress_until(current_alpha):
                signal = progress.signal
                signal.timestamp = relative_rate_inv(progress.signal_alpha)
                if isinstance(signal, TimelineStartSignal):
                    #signal.absolute_rate = RateUtils.compose(
                    #    RateUtils.adjust(signal.absolute_rate, lag_time=progress.timeline_start_alpha),
                    #    relative_rate
                    #)
                    signal.composed_updater = compose_updater(
                        signal.composed_updater,
                        RateUtils.adjust(relative_rate, lag_alpha=progress.timeline_start_alpha)
                    )
                yield signal
                #if (absolute_rate := signal.absolute_rate) is not None:
                #    new_absolute_rate = RateUtils.compose(
                #        RateUtils.adjust(absolute_rate, lag_time=signal_timeline_start_alpha),
                #        relative_rate
                #    )
                #else:
                #    new_absolute_rate = None

                #yield TimelineSignal(
                #    timestamp=relative_rate_inv(signal_alpha),
                #    animation=signal.animation,
                #    absolute_rate=new_absolute_rate,
                #    timeline_state=signal.timeline_state
                #)

            #while manager.is_not_empty():
            #    signal_timeline, timeline_start_alpha, signal = manager.get_next_signal_timeline_item()
            #    next_alpha = timeline_start_alpha + signal.timestamp
            #    if next_alpha > current_alpha:
            #        break

            #    if (absolute_rate := signal.absolute_rate) is not None:
            #        new_absolute_rate = RateUtils.compose(
            #            RateUtils.adjust(absolute_rate, lag_time=timeline_start_alpha),
            #            relative_rate
            #        )
            #    else:
            #        new_absolute_rate = None

            #    yield TimelineSignal(
            #        timestamp=relative_rate_inv(next_alpha),
            #        animation=signal.animation,
            #        absolute_rate=new_absolute_rate,
            #        timeline_state=signal.timeline_state
            #    )
            #    manager.advance_to_next_signal(signal_timeline)

            #yield TimelineSignal(
            #    timestamp=relative_rate_inv(current_alpha),
            #    animation=self,
            #    absolute_rate=None,
            #    timeline_state=TimelineState.AWAIT
            #)
            yield TimelineAwaitSignal(
                timestamp=current_time
                #animation_id=animation_id
            )

            if early_break:
                # TODO: yield TimelineState.STOP from rest signals in manager
                for progress in manager.iter_timeline_progress_until(None):  #
                    yield TimelineStopSignal(
                        timestamp=current_time,
                        animation_id=progress.signal.animation_id
                    )
                    #yield TimelineSignal(  #
                    #    timestamp=relative_rate_inv(current_alpha),  #
                    #    animation=signal.animation,  #
                    #    absolute_rate=None,  #
                    #    timeline_state=TimelineState.STOP  #
                    #)  #
                break

        yield TimelineStopSignal(
            timestamp=current_time,
            animation_id=animation_id
        )
        #yield TimelineSignal(
        #    timestamp=relative_rate_inv(current_alpha),
        #    animation=self,
        #    absolute_rate=None,
        #    timeline_state=TimelineState.STOP
        #)

    @property
    def _play_run_time(self) -> float:
        return run_time if (run_time := self._run_time) is not None else 0.0

    def _get_bound_scene(self) -> "Scene":
        return Toplevel.get_scene_by_animation_id(self._id)

    def prepare(
        self,
        *animations: "Animation"
    ) -> None:
        bound_scene = self._get_bound_scene()
        for animation in animations:
            Toplevel.bind_animation_id_to_scene(animation._id, bound_scene)
            #assert not animation._is_prepared_flag
            #animation._is_prepared_flag = True
            #if animation._scene_ref is None:
            #    animation._scene_ref = self._scene_ref
        self._new_children.extend(animations)

    async def wait(
        self,
        delta_alpha: float = 1.0
    ) -> None:
        self._delta_alpha = delta_alpha
        await asyncio.sleep(0.0)

    async def play(
        self,
        animation: "Animation"
    ) -> None:
        self.prepare(animation)
        await self.wait(animation._play_run_time)

    async def wait_forever(self) -> None:
        # Used for infinite animation.
        while True:
            await self.wait()

    @abstractmethod
    async def timeline(self) -> None:
        pass

    # Access the scene the animation is operated on.
    # Always accessible in the body of `timeline()` method.
    @property
    def scene(self) -> "Scene":
        return Toplevel.get_scene()
        #assert (scene_ref := self._scene_ref) is not None
        #assert (scene := scene_ref()) is not None
        #return scene
