from dataclasses import dataclass
from enum import Enum
#from functools import partial
#from enum import Enum
#import asyncio
#import operator as op
from typing import (
    Callable,
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
    absolute_rate: Callable[[float], float]


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TimelineState:
    timestamp: float
    updater_item: UpdaterItem
    #updater: Callable[[float], None] | None
    #absolute_rate: Callable[[float], float]
    verb: UpdaterItemVerb
    #removal_updater_items: list[UpdaterItem]
    #additional_updater_items: list[UpdaterItem]


#@dataclass(
#    kw_only=True,
#    slots=True
#)
class TimelineManager:
    __slots__ = (
        "timeline_start_alpha_dict",
        "state_dict"
        #"next_alpha_dict"
    )

    def __init__(self) -> None:
        super().__init__()
        self.timeline_start_alpha_dict: dict[Iterator[TimelineState], float] = {}
        self.state_dict: dict[Iterator[TimelineState], TimelineState] = {}
        #self.next_alpha_dict: dict[Iterator[TimelineState], float] = {}
        #self._absolute_timeline: Iterator[TimelineState] = timeline
        #self._start_alpha: float = start_alpha
        #updater_item: UpdaterItem
        #self._state: TimelineState | None = None
        #self.advance_state()
        #try:
        #    self._state = next(timeline)
        #except StopIteration:
        #    self._state = None

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
        #self.next_alpha_dict[timeline] = start_alpha + state.timestamp

    def advance_state(
        self,
        timeline: Iterator[TimelineState]
    ) -> None:
        try:
            state = next(timeline)
        except StopIteration:
            self.timeline_start_alpha_dict.pop(timeline)
            self.state_dict.pop(timeline)
            #self.next_alpha_dict.pop(timeline)
            return
        self.state_dict[timeline] = state
        #self.next_alpha_dict[timeline] = self.timeline_start_alpha_dict[timeline] + state.timestamp

    def is_not_empty(self) -> bool:
        return bool(self.state_dict)

    def get_next_timeline_item(self) -> tuple[Iterator[TimelineState], float, TimelineState]:
        timeline_start_alpha_dict = self.timeline_start_alpha_dict
        state_dict = self.state_dict

        def get_next_alpha(
            timeline: Iterator[TimelineState]
        ) -> float:
            return timeline_start_alpha_dict[timeline] + state_dict[timeline].timestamp

        #next_alpha_dict = self.next_alpha_dict
        timeline = min(state_dict, key=get_next_alpha)
        #min_stop_alpha = min(
        #    child_state.stop_time
        #    for child_state in awaiting_timelines.values()
        #)
        #child_timestamp = relative_rate_inv(timeline_to_state[timeline].timestamp)
        #if child_timestamp > current_time:
        #    break
        #next_alpha = get_next_alpha(timeline)
        #start_alpha = timeline_start_alpha_dict[timeline]
        #state = state_dict[timeline]
        return timeline, timeline_start_alpha_dict[timeline], state_dict[timeline]

    #def advance_state(self) -> None:
    #    try:
    #        self._state = next(self._absolute_timeline)
    #    except StopIteration:
    #        self._state = None




#UpdaterItemT = tuple[Callable[[float], None], Callable[[float], float]]
#_AbsoluteTimelineT = Generator[TimelineState, None, None]


class Animation:
    __slots__ = (
        #"_start_time",
        "_run_time",
        #"_lag_time",
        "_relative_rate",
        #"_rate_inv",
        "_updater",
        #"_timeline",
        #"_timeline",
        #"_current_alpha",
        "_new_children"
        #"_absolute_rate",
        #"_all_children"
        #"_rest_wait_time"
    )

    def __init__(
        self,
        # If provided, the animation will be clipped from right at `run_time`.
        # `parent.play(self)` will call `parent.wait(run_time)` that covers this animation.
        run_time: float | None = None,
        # Adjusts the starting position of the animation on its parent's timeline.
        #lag_time: float = 0.0,
        # `[0.0, run_time] -> [0.0, +infty), time |-> alpha`
        # Must be an increasing function.
        relative_rate: Callable[[float], float] = RateUtils.linear,
        # Update continuously (per frame).
        updater: Callable[[float], None] | None = None
    ) -> None:
        super().__init__()
        #if timeline is None:
        #    timeline = self.timeline()
        #assert start_time >= 0.0
        assert run_time is None or run_time >= 0.0
        #assert lag_time >= 0.0
        #assert relative_rate(0.0) >= 0.0
        #self._start_time: float = start_time
        self._run_time: float | None = run_time
        #self._lag_time: float = lag_time
        self._relative_rate: Callable[[float], float] = relative_rate
        #self._rate_inv: Callable[[float], float] = RateUtils.inverse(rate)
        self._updater: Callable[[float], None] | None = updater
        #self._updater: Callable[[float], None] | None = updater
        #self._timeline: Generator[float, None, None] = timeline
        # General updater.
        #self._timeline: Generator[float, None, None] = self.timeline()
        #self._current_alpha: float = 0.0

        # For internal usage.
        self._new_children: list[Animation] = []
        #self._rest_wait_time: float | None = None

        #self._absolute_rate: Callable[[float], float] = NotImplemented
        #self._all_children: list[Animation] = []

    #def updater(
    #    self,
    #    alpha: float
    #) -> None:
    #    pass

    # Yield `delta_alpha` values.
    def timeline(self) -> Iterator[float]:
        raise StopIteration

    #@property
    #def _max_time(self) -> float | None:
    #    if self._run_time is None:
    #        return None
    #    return self._lag_time + self._run_time

    #@property
    #def _rate_inv(self) -> Callable[[float], float]:
    #    func = self._rate

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
    #    rate: Callable[[float], float] | None = None
    #) -> None:
    #    assert stop_time is None or stop_time >= start_time
    #    if rate is None:
    #        rate = RateUtils.linear

    #    def time_animate_func(
    #        t0: float,
    #        t: float
    #    ) -> None:
    #        alpha_animate_func(rate(t0), rate(t))

    #    def alpha_to_time(
    #        alpha: float
    #    ) -> float:
    #        t = RateUtils.inverse(rate, alpha)
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

    def _absolute_timeline(self) -> Iterator[TimelineState]:

        #def lag_rate(
        #    #rate: Callable[[float], float],
        #    lag_time: float
        #) -> Callable[[float], float]:

        #    def result(
        #        t: float
        #    ) -> float:
        #        return t - lag_time

        #    return result

        #start_time = self._start_time
        #run_time = self._stop_time
        #lag_time = self._lag_time
        #rate_lag_time = lag_time - start_time
        relative_rate = self._relative_rate
        relative_rate_inv = RateUtils.inverse(relative_rate)
        run_alpha = relative_rate(self._run_time) if self._run_time is not None else None
        #current_time = 0.0
        current_alpha = relative_rate(0.0)
        assert current_alpha >= 0.0
        #timeline = self._timeline
        #updater = self._updater
        #timeline = self.timeline()

        #accumuted_timeline_to_updater_item: dict[_AbsoluteTimelineT, UpdaterItem] = {}
        #absolute_timelines: dict[_AbsoluteTimelineT, UpdaterItem] = {}

        #awaiting_absolute_timelines: dict[Iterator[TimelineState], TimelineState] = {}
        #processing_absolute_timelines: list[Iterator[TimelineState]] = []
        #absolute_timeline_infos: list[AbsoluteTimelineInfo] = []
        #absolute_timeline_to_start_alpha: dict[Iterator[TimelineState], float] = {}
        #absolute_timeline_info: dict[Iterator[TimelineState], AbsoluteTimelineInfo] = {}
        #absolute_timeline_to_updater_item: dict[Iterator[TimelineState], UpdaterItem] = {}
        #absolute_timeline_to_state: dict[Iterator[TimelineState], TimelineState] = {}
        manager = TimelineManager()
        updater_item_convert_dict: dict[UpdaterItem, UpdaterItem] = {}

        #def get_alpha_from_absolute_timeline(
        #    absolute_timeline: Iterator[TimelineState]
        #) -> float:
        #    return absolute_timeline_to_start_alpha[absolute_timeline] + absolute_timeline_to_state[absolute_timeline].timestamp

        #absolute_rate_cache: dict[_AbsoluteTimelineT, Callable[[float], float]] = {}
        #updater_items: list[UpdaterItem] = []

        #yield TimelineState(
        #    stop_time=lag_time,
        #    removal_updater_items=[],
        #    additional_updater_items=[]
        #)
        self_updater_item = UpdaterItem(
            updater=self._updater,
            absolute_rate=relative_rate
        )
        yield TimelineState(
            timestamp=0.0,
            updater_item=self_updater_item,
            verb=UpdaterItemVerb.APPEND
        )
        for wait_delta_alpha in self.timeline():
            #try:
            #    wait_delta_alpha = timeline.send(None)
            #except StopIteration:
            #    break

            #new_absolute_timelines = {
            #    child._absolute_timeline(): UpdaterItem(
            #        updater=child._updater,
            #        absolute_rate=RateUtils.compose(
            #            child._rate,
            #            lag_rate(current_time),
            #            rate
            #        )
            #    )
            #    for child in self._new_children
            #}
            #absolute_timelines.update(new_absolute_timelines)
            #processing_absolute_timelines.extend(
            #    child._absolute_timeline()
            #    for child in self._new_children
            #)
            #self._new_children.clear()
            while self._new_children:
                child = self._new_children.pop(0)
                #absolute_timeline = child._absolute_timeline()
                #try:
                #    absolute_timeline_info = AbsoluteTimelineInfo(
                #        absolute_timeline=absolute_timeline,
                #        start_alpha=current_alpha
                #    )
                #except StopIteration:
                #    continue
                manager.add_timeline(
                    timeline=child._absolute_timeline(),
                    start_alpha=current_alpha
                )
                #absolute_timeline_infos.append(AbsoluteTimelineInfo(
                #    absolute_timeline=child._absolute_timeline(),
                #    start_alpha=current_alpha
                #))
                #absolute_timeline_to_start_alpha[absolute_timeline] = current_alpha
                #absolute_timeline_to_state[absolute_timeline] = state
                #absolute_timeline_info[absolute_timeline] = AbsoluteTimelineInfo(
                #    start_alpha=current_alpha,
                #    updater_item=UpdaterItem(
                #        updater=state.updater_item.updater,
                #        absolute_rate=RateUtils.compose(
                #            RateUtils.adjust(state.updater_item.absolute_rate, lag_time=current_alpha),
                #            relative_rate
                #            #RateUtils.adjust(rate, lag_time=lag_time - start_time)
                #            #lag_rate(child_timestamp),
                #            #rate
                #        )
                #    ),
                #    state=state
                #)
                #absolute_timeline_to_start_alpha[absolute_timeline] = current_alpha
                #absolute_timeline_to_updater_item[absolute_timeline] = 
                #)
                #absolute_timeline_to_state[absolute_timeline] = state



            assert wait_delta_alpha >= 0.0
            #current_time = relative_rate_inv(relative_rate(current_time) + wait_delta_alpha)
            current_alpha += wait_delta_alpha
            if run_alpha is not None and current_alpha > run_alpha:
                early_break = True
                current_alpha = run_alpha
            else:
                early_break = False
            #current_time = relative_rate_inv(current_alpha)
            #accumuted_timeline_to_updater_item.update(new_items)
            #absolute_timelines.update(new_absolute_timelines)
            #processing_absolute_timelines.extend(new_absolute_timelines)
            #yield TimelineState(
            #    stop_time=lag_time + current_time - start_time,
            #    removal_updater_items=list(new_absolute_timelines.values()),
            #    additional_updater_items=[]
            #)
            #self._new_children.clear()
            #next_time
            #updater_items: list[UpdaterItem] = []
            #if updater is not None:
            #    updater_items.append(UpdaterItem(
            #        updater=updater,
            #        absolute_rate=rate
            #    ))
            #removal_updater_items: list[UpdaterItem] = []
            #additional_updater_items: list[UpdaterItem] = list(new_timeline_items.values())
            #self._new_children.clear()

            while manager.is_not_empty():
                #pending_absolute_timelines: dict[_AbsoluteTimelineT, TimelineState] = []
                #for child_absolute_timeline in processing_absolute_timelines:
                #    try:
                #        child_state = next(child_absolute_timeline)
                #    except StopIteration:
                #        continue
                #        #absolute_timelines.pop(child_absolute_timeline)
                #        #removal_updater_items.append(accumuted_timeline_to_updater_item.pop(child_absolute_timeline))
                #        #continue
                #        #absolute_timelines[child_absolute_timeline] = child_state
                #    #pending_absolute_timelines[child_absolute_timeline] = child_state
                #    awaiting_absolute_timelines[child_absolute_timeline] = child_state

                #processing_absolute_timelines.clear()
                #absolute_timelines.clear()
                #absolute_timelines.update(pending_absolute_timelines)

                        #for child_updater_item in child_updater_items:
                        #    child_updater, child_absolute_rate = child_updater_item
                        #    if (absolute_rate := absolute_rate_cache.get(child_absolute_timeline)) is None:
                        #        absolute_rate = RateUtils.compose(
                        #            child_absolute_rate,
                        #            lag_rate(current_time),
                        #            rate
                        #        )
                        #        absolute_rate_cache[child_absolute_timeline] = absolute_rate
                        #    updater_items.append((child_updater, absolute_rate))
                        #removal_updater_items.extend(child_removal_updater_items)
                        #additional_updater_items.extend(child_additional_updater_items)
                        #child_rest_time = child_time
                        #child_state = TimelineState(
                        #    stop_time=child_time,
                        #    removal_updater_items=child_removal_updater_items,
                        #    additional_updater_items=child_additional_updater_items
                        #)

                    #if child_state.stop_time <= new_alpha:
                    #    children_states.append(child_state)

                #if manager.is_empty():
                #    break
                timeline, timeline_start_alpha, state = manager.get_next_timeline_item()

                #for absolute_timeline_info in absolute_timeline_infos[:]:
                #    if absolute_timeline_info._state is None:
                #        absolute_timeline_infos.remove(absolute_timeline_info)

                #if not absolute_timeline_info:
                #    break
                #next_alpha_dict = {
                #    absolute_timeline: info.start_alpha + info.state.timestamp
                #    for absolute_timeline, info in absolute_timeline_info.items()
                #}
                #absolute_timeline = min(absolute_timeline_info, key=next_alpha_dict.__getitem__)
                ##min_stop_alpha = min(
                ##    child_state.stop_time
                ##    for child_state in awaiting_absolute_timelines.values()
                ##)
                ##child_timestamp = relative_rate_inv(absolute_timeline_to_state[absolute_timeline].timestamp)
                ##if child_timestamp > current_time:
                ##    break
                #next_alpha = next_alpha_dict[absolute_timeline]
                next_alpha = timeline_start_alpha + state.timestamp
                if next_alpha > current_alpha:
                    break

                #state = absolute_timeline_to_state[absolute_timeline]
                updater_item = state.updater_item
                verb = state.verb
                if verb == UpdaterItemVerb.APPEND:
                    new_updater_item = UpdaterItem(
                        updater=updater_item.updater,
                        absolute_rate=RateUtils.compose(
                            #updater_item.absolute_rate,
                            #rate
                            RateUtils.adjust(updater_item.absolute_rate, lag_time=timeline_start_alpha),
                            relative_rate
                            #RateUtils.adjust(rate, lag_time=lag_time - start_time)
                            #lag_rate(child_timestamp),
                            #rate
                        )
                    )
                    updater_item_convert_dict[updater_item] = new_updater_item
                else:
                    new_updater_item = updater_item_convert_dict.pop(updater_item)
                yield TimelineState(
                    timestamp=relative_rate_inv(next_alpha),
                    updater_item=new_updater_item,
                    verb=verb
                )

                manager.advance_state(timeline)
                #try:
                #    absolute_timeline_info[absolute_timeline].state = next(absolute_timeline)
                #except StopIteration:
                #    absolute_timeline_info.pop(absolute_timeline)
                    #absolute_timeline_to_updater_item.pop(absolute_timeline)
                    #absolute_timeline_to_state.pop(absolute_timeline)
                    #continue
                #absolute_timeline_to_state[absolute_timeline] = state


                #child_state = awaiting_absolute_timelines.pop(child_absolute_timeline)
                #processing_absolute_timelines.append(child_absolute_timeline)

                #if (child_updater_item := updater_item_convert_dict.pop(child_state.updater_item, None)) is None:
                #    assert child_state.verb == UpdaterItemVerb.APPEND
                #    child_updater_item = UpdaterItem(
                #        updater=child_state.updater_item.updater,
                #        absolute_rate=RateUtils.compose(
                #            child_state.updater_item.absolute_rate,
                #            rate
                #            #RateUtils.adjust(rate, lag_time=lag_time - start_time)
                #            #lag_rate(child_timestamp),
                #            #rate
                #        )
                #    )
                #    updater_item_convert_dict[child_state.updater_item] = child_updater_item
                #else:
                #    assert child_state.verb == UpdaterItemVerb.REMOVE
                #yield TimelineState(
                #    timestamp=child_timestamp,
                #    updater_item=child_updater_item,
                #    verb=child_state.verb
                #)
                #for child_absolute_timeline, child_state in awaiting_absolute_timelines.copy().items():
                #    if min_stop_alpha == child_state.stop_time:
                #        awaiting_absolute_timelines.pop(child_absolute_timeline)
                #        processing_absolute_timelines.append(child_absolute_timeline)

            #current_time = new_time
            if early_break:
                break

        yield TimelineState(
            timestamp=relative_rate_inv(current_alpha),
            updater_item=self_updater_item,
            verb=UpdaterItemVerb.REMOVE
        )
        #yield TimelineState(
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
        #delta_time = self._rate_inv(self._rate(self._current_time) + delta_alpha) - self._current_time
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
            run_time
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
    #    #await self.wait(wait_time),
