__all__ = [
    "Animation"
    #"RegroupItem",
    #"RegroupVerb"
]


from dataclasses import dataclass
from enum import Enum
from functools import partial
#from enum import Enum
#import asyncio
#import operator as op
from typing import (
    Callable,
    Generator,
    Iterator
)

#import warnings

#import numpy as np
#import scipy

#from ..mobjects.mobject import Mobject
from ..utils.rate import RateUtils


#class RegroupVerb(Enum):
#    ADD = 1
#    BECOMES = 0
#    DISCARD = -1


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class RegroupItem:
#    mobjects: Mobject | Iterable[Mobject]
#    verb: RegroupVerb
#    targets: Mobject | Iterable[Mobject]


class UpdaterItemVerb(Enum):
    APPEND = 1
    REMOVE = -1


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class UpdaterItem:
    updater: Callable[[float], None] | None
    absolute_rate_func: Callable[[float], float]


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class AccumulatedTimelineState:
    updater_item: UpdaterItem
    #updater: Callable[[float], None] | None
    #absolute_rate_func: Callable[[float], float]
    timestamp: float
    verb: UpdaterItemVerb
    #removal_updater_items: list[UpdaterItem]
    #additional_updater_items: list[UpdaterItem]


#UpdaterItemT = tuple[Callable[[float], None], Callable[[float], float]]
AccumulatedTimelineT = Generator[AccumulatedTimelineState, None, None]


class Animation:
    __slots__ = (
        "_start_time",
        "_run_time",
        "_lag_time",
        "_rate_func",
        #"_rate_func_inv",
        "_updater",
        "_timeline",
        #"_timeline",
        #"_current_alpha",
        "_new_children",
        #"_absolute_rate_func",
        #"_all_children"
        #"_rest_wait_time"
    )

    def __init__(
        self,
        # The animation will be clipped from left at `start_time`.
        start_time: float = 0.0,
        # The animation will be clipped from right at `start_time + run_time`.
        # If provided, `parent.play(self)` will call `wait()`
        # by a suitable time span covering this animation.
        run_time: float | None = None,
        # Adjusts the starting position of the animation on its parent's timeline.
        lag_time: float = 0.0,
        # `(0, run_time) |-> (0, run_alpha)`
        # Must be an increasing function.
        rate_func: Callable[[float], float] = RateUtils.linear,
        # Update continuously (per frame). Called on `alpha`.
        updater: Callable[[float], None] | None = None,
        # Yield `delta_alpha` values.
        timeline: Generator[float, None, None] | None = None
    ) -> None:
        super().__init__()
        if timeline is None:
            timeline = self.timeline()
        assert start_time >= 0.0
        assert run_time is None or run_time >= 0.0
        assert lag_time >= 0.0
        self._start_time: float = start_time
        self._run_time: float | None = run_time
        self._lag_time: float = lag_time
        self._rate_func: Callable[[float], float] = rate_func
        #self._rate_func_inv: Callable[[float], float] = RateUtils.inverse(rate_func)
        self._updater: Callable[[float], None] | None = updater
        self._timeline: Generator[float, None, None] = timeline
        # General updater.
        #self._timeline: Generator[float, None, None] = self.timeline()
        #self._current_alpha: float = 0.0

        # For internal usage.
        self._new_children: list[Animation] = []
        #self._rest_wait_time: float | None = None

        #self._absolute_rate_func: Callable[[float], float] = NotImplemented
        #self._all_children: list[Animation] = []

    #@property
    #def _suitable_play_wait_time(self) -> float | None:
    #    if self._run_time is None:
    #        return None
    #    return self._lag_time + self._run_time

    #@property
    #def _rate_func_inv(self) -> Callable[[float], float]:
    #    func = self._rate_func

    #    def inverse(
    #        y: float
    #    ) -> float:
    #        for x0 in np.linspace(0.0, 1.0, 5):
    #            optimize_result = scipy.optimize.root(lambda x: func(x) - y, x0)
    #            if optimize_result.success:
    #                return float(optimize_result.x)
    #        raise ValueError

    #    return inverse

    #__slots__ = (
    #    "_time_animate_func",
    #    "_time_regroup_items",
    #    "_start_time",
    #    "_stop_time"
    #)

    #def __init__(
    #    self,
    #    *,
    #    # Two arguments provided are `(alpha_0, alpha)`.
    #    alpha_animate_func: Callable[[float, float], None],
    #    #alpha_regroup_items: list[tuple[float, RegroupItem]],
    #    start_time: float,
    #    stop_time: float | None,
    #    # `time |-> alpha`
    #    rate_func: Callable[[float], float] | None = None
    #) -> None:
    #    assert stop_time is None or stop_time >= start_time
    #    if rate_func is None:
    #        rate_func = RateUtils.linear

    #    def time_animate_func(
    #        t0: float,
    #        t: float
    #    ) -> None:
    #        alpha_animate_func(rate_func(t0), rate_func(t))

    #    def alpha_to_time(
    #        alpha: float
    #    ) -> float:
    #        t = RateUtils.inverse(rate_func, alpha)
    #        if stop_time is not None and t > stop_time:
    #            if not np.isclose(t, stop_time):
    #                warnings.warn("`time_regroup_items` is not within `(start_time, stop_time)`")
    #            t = stop_time
    #        return t

    #    self._time_animate_func: Callable[[float, float], None] = time_animate_func
    #    #self._time_regroup_items: list[tuple[float, RegroupItem]] = [
    #    #    (alpha_to_time(alpha), regroup_item)
    #    #    for alpha, regroup_item in alpha_regroup_items
    #    #]
    #    self._start_time: float = start_time
    #    self._stop_time: float | None = stop_time

    #def _iter_animation_descendants(self) -> "Iterator[Animation]":
    #    yield self
    #    for child in self._children:
    #        yield from child._iter_animation_descendants()

    def _accumulated_timeline(self) -> AccumulatedTimelineT:

        def lag_rate_func(
            #rate_func: Callable[[float], float],
            lag_time: float
        ) -> Callable[[float], float]:

            def result(
                t: float
            ) -> float:
                return t - lag_time

            return result

        start_time = self._start_time
        run_time = self._run_time
        stop_time = start_time + run_time if run_time is not None else None
        lag_time = self._lag_time
        current_time = start_time
        rate_func = self._rate_func
        rate_func_inv = RateUtils.inverse(rate_func)
        timeline = self._timeline
        #updater = self._updater
        #timeline = self.timeline()

        #accumuted_timeline_to_updater_item: dict[AccumulatedTimelineT, UpdaterItem] = {}
        #accumulated_timelines: dict[AccumulatedTimelineT, UpdaterItem] = {}
        awaiting_accumulated_timelines: dict[AccumulatedTimelineT, AccumulatedTimelineState] = {}
        processing_accumulated_timelines: list[AccumulatedTimelineT] = []
        updater_item_convert_dict: dict[UpdaterItem, UpdaterItem] = {}
        #absolute_rate_func_cache: dict[AccumulatedTimelineT, Callable[[float], float]] = {}
        #updater_items: list[UpdaterItem] = []

        #yield AccumulatedTimelineState(
        #    stop_time=lag_time,
        #    removal_updater_items=[],
        #    additional_updater_items=[]
        #)
        updater_item = UpdaterItem(
            updater=self._updater,
            absolute_rate_func=rate_func
        )
        yield AccumulatedTimelineState(
            updater_item=updater_item,
            timestamp=lag_time,
            verb=UpdaterItemVerb.APPEND
        )
        while True:
            try:
                wait_delta_alpha = timeline.send(None)
            except StopIteration:
                break
            assert wait_delta_alpha >= 0.0
            new_time = rate_func_inv(rate_func(current_time) + wait_delta_alpha)
            if stop_time is not None and new_time > stop_time:
                early_break = True
                new_time = stop_time
            else:
                early_break = False

            #new_accumulated_timelines = {
            #    child._accumulated_timeline(): UpdaterItem(
            #        updater=child._updater,
            #        absolute_rate_func=RateUtils.compose(
            #            child._rate_func,
            #            lag_rate_func(current_time),
            #            rate_func
            #        )
            #    )
            #    for child in self._new_children
            #}
            #accumulated_timelines.update(new_accumulated_timelines)
            processing_accumulated_timelines.extend(
                child._accumulated_timeline()
                for child in self._new_children
            )
            self._new_children.clear()
            #accumuted_timeline_to_updater_item.update(new_items)
            #accumulated_timelines.update(new_accumulated_timelines)
            #processing_accumulated_timelines.extend(new_accumulated_timelines)
            #yield AccumulatedTimelineState(
            #    stop_time=lag_time + current_time - start_time,
            #    removal_updater_items=list(new_accumulated_timelines.values()),
            #    additional_updater_items=[]
            #)
            #self._new_children.clear()
            #next_time
            #updater_items: list[UpdaterItem] = []
            #if updater is not None:
            #    updater_items.append(UpdaterItem(
            #        updater=updater,
            #        absolute_rate_func=rate_func
            #    ))
            #removal_updater_items: list[UpdaterItem] = []
            #additional_updater_items: list[UpdaterItem] = list(new_timeline_items.values())
            #self._new_children.clear()

            while True:
                #pending_accumulated_timelines: dict[AccumulatedTimelineT, AccumulatedTimelineState] = []
                for child_accumulated_timeline in processing_accumulated_timelines:
                    try:
                        child_state = child_accumulated_timeline.send(None)
                    except StopIteration:
                        continue
                        #accumulated_timelines.pop(child_accumulated_timeline)
                        #removal_updater_items.append(accumuted_timeline_to_updater_item.pop(child_accumulated_timeline))
                        #continue
                        #accumulated_timelines[child_accumulated_timeline] = child_state
                    #pending_accumulated_timelines[child_accumulated_timeline] = child_state
                    awaiting_accumulated_timelines[child_accumulated_timeline] = child_state

                processing_accumulated_timelines.clear()
                #accumulated_timelines.clear()
                #accumulated_timelines.update(pending_accumulated_timelines)

                        #for child_updater_item in child_updater_items:
                        #    child_updater, child_absolute_rate_func = child_updater_item
                        #    if (absolute_rate_func := absolute_rate_func_cache.get(child_accumulated_timeline)) is None:
                        #        absolute_rate_func = RateUtils.compose(
                        #            child_absolute_rate_func,
                        #            lag_rate_func(current_time),
                        #            rate_func
                        #        )
                        #        absolute_rate_func_cache[child_accumulated_timeline] = absolute_rate_func
                        #    updater_items.append((child_updater, absolute_rate_func))
                        #removal_updater_items.extend(child_removal_updater_items)
                        #additional_updater_items.extend(child_additional_updater_items)
                        #child_rest_time = child_time
                        #child_state = AccumulatedTimelineState(
                        #    stop_time=child_time,
                        #    removal_updater_items=child_removal_updater_items,
                        #    additional_updater_items=child_additional_updater_items
                        #)

                    #if child_state.stop_time <= new_alpha:
                    #    children_states.append(child_state)

                if not awaiting_accumulated_timelines:
                    break
                target_accumulated_timeline = min(
                    awaiting_accumulated_timelines,
                    key=partial(
                        lambda child_accumulated_timeline: awaiting_accumulated_timelines[child_accumulated_timeline].timestamp
                    )
                )
                #min_stop_alpha = min(
                #    child_state.stop_time
                #    for child_state in awaiting_accumulated_timelines.values()
                #)
                child_stop_time = rate_func_inv(awaiting_accumulated_timelines[target_accumulated_timeline].timestamp)
                if child_stop_time > new_time:
                    break
                child_state = awaiting_accumulated_timelines.pop(target_accumulated_timeline)
                processing_accumulated_timelines.append(target_accumulated_timeline)

                if (child_updater_item := updater_item_convert_dict.pop(child_state.updater_item)) is None:
                    child_updater_item = UpdaterItem(
                        updater=child_state.updater_item.updater,
                        absolute_rate_func=RateUtils.compose(
                            child_state.updater_item.absolute_rate_func,
                            lag_rate_func(child_stop_time),
                            rate_func
                        )
                    )
                    updater_item_convert_dict[child_state.updater_item] = child_updater_item
                yield AccumulatedTimelineState(
                    updater_item=child_updater_item,
                    timestamp=lag_time + child_stop_time - start_time,
                    verb=child_state.verb
                )
                #for child_accumulated_timeline, child_state in awaiting_accumulated_timelines.copy().items():
                #    if min_stop_alpha == child_state.stop_time:
                #        awaiting_accumulated_timelines.pop(child_accumulated_timeline)
                #        processing_accumulated_timelines.append(child_accumulated_timeline)

            if early_break:
                break
            current_time = new_time

        yield AccumulatedTimelineState(
            updater_item=updater_item,
            timestamp=lag_time + current_time - start_time,
            verb=UpdaterItemVerb.REMOVE
        )
        #yield AccumulatedTimelineState(
        #    stop_time=lag_time + current_time - start_time,
        #    updater_item=updater_item,
        #    verb=UpdaterItemVerb.REMOVE
        #)

    def prepare(
        self,
        *animations: "Animation"
    ) -> None:
        #for animation in animations:
        #    self._animation_dict[animation] = 0.0
        self._new_children.extend(animations)
        #return self

    def wait(
        self,
        delta_alpha: float = 1.0
    ) -> Iterator[float]:
        #self._current_alpha += delta_alpha
        yield delta_alpha
        #delta_time = self._rate_func_inv(self._rate_func(self._current_time) + delta_alpha) - self._current_time
        #self._current_time += delta_time
        #yield delta_time

    #def wait(
    #    self,
    #    t: float = 1.0
    #) -> None:
    #    self._rest_wait_time = t
    #    await asyncio.sleep(0.0)
        #assert t >= 0.0
        #frames = t * ConfigSingleton().rendering.fps
        #start_frame_floating_index = self._frame_floating_index
        #stop_frame_floating_index = start_frame_floating_index + frames
        #self._frame_floating_index = stop_frame_floating_index
        #frame_range, reaches_end = self._find_frame_range(start_frame_floating_index, stop_frame_floating_index)

        #if not frame_range:
        #    self._update_frames(frames)
        #else:
        #    self._update_frames(frame_range.start - start_frame_floating_index)
        #    if self._previous_frame_rendering_timestamp is None:
        #        self._process_rendering(render_to_video=True)
        #    for _ in frame_range[:-1]:
        #        self._update_frames(1)
        #        self._process_rendering(render_to_video=True)
        #    self._update_frames(stop_frame_floating_index - (frame_range.stop - 1))

        #if reaches_end:
        #    raise EndSceneException()

    def play(
        self,
        *animations: "Animation"
    ) -> Iterator[float]:
        self.prepare(*animations)
        delta_alpha = max((
            animation._lag_time + run_time
            for animation in animations
            if (run_time := animation._run_time) is not None
        ), default=0.0)
        yield from self.wait(delta_alpha)

    #async def play(
    #    self,
    #    *animations: "Animation"
    #) -> None:
    #    self.prepare(*animations)
    #    for animation in animations:
    #        await animation.timeline()
    #    #try:
    #    #    wait_time = max(t for animation in animations if (t := animation._stop_time) is not None)
    #    #except ValueError:
    #    #    wait_time = 0.0
    #    #await self.wait(wait_time)

    def timeline(self) -> Generator[float, None, None]:
        raise StopIteration
