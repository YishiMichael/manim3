#from dataclasses import dataclass
#from typing import TYPE_CHECKING, Coroutine

#from .conditions.condition import Condition
#from .rates.rate import Rate

#if TYPE_CHECKING:
#    from .animation import Animation




    #def __init__(
    #    self,
    #    schedule_info: ScheduleInfo
    #) -> None:
    #    super().__init__()
    #    self._schedule_info: ScheduleInfo = schedule_info

    #@abstractmethod
    #def _state_progress(
    #    self,
    #    animation: "Animation",
    #    timestamp: float
    #) -> "AnimatingState | None":
    #    pass

    #def _state_progress(
    #    self,
    #    animation: "Animation"
    #) -> AnimatingState | None:
    #    if self._launch_condition.judge():
    #        return OnAnimating(self._schedule_info, animation)
    #    return None
    #__slots__ = (
    #    "_timeline_coroutine",
    #    "_absolute_rate",
    #    "_terminate_condition",
    #    "_progress_condition",
    #    "_children"
    #)

    #def __init__(
    #    self,
    #    timeline_coroutine: Coroutine[None, None, None],
    #    absolute_rate: Rate,
    #    terminate_condition: Condition,
    #    progress_condition: Condition,
    #    children: list[Animation]
    #) -> None:
    #    super().__init__()
    #    self._timeline_coroutine: Coroutine[None, None, None] = timeline_coroutine
    #    self._absolute_rate: Rate = AbsoluteRate(
    #        parent_absolute_rate=schedule_info.parent_absolute_rate,
    #        relative_rate=schedule_info.relative_rate
    #    )
    #    self._terminate_condition: Condition = schedule_info.terminate_condition
    #    self._progress_condition: Condition = Always()
    #    self._children: list[Animation] = []

    #def _state_progress(
    #    self,
    #    animation: "Animation",
    #    timestamp: float
    #) -> AnimatingState | None:
    #    animation.update(self._absolute_rate.at(timestamp))
    #    while not self._terminate_condition.judge():
    #        for child in self._children[:]:
    #            child._progress()
    #            if isinstance(child._animating_state, AfterAnimating):
    #                self._children.remove(child)
    #        assert (progress_condition := self._progress_condition) is not None
    #        if not progress_condition.judge():
    #            return None
    #        try:
    #            self._timeline_coroutine.send(None)
    #        except StopIteration:
    #            break
    #    return AfterAnimating(self._schedule_info, animation, timestamp)
    #__slots__ = ()

    #def _state_progress(
    #    self,
    #    animation: "Animation",
    #    timestamp: float
    #) -> AnimatingState | None:
    #    return None
