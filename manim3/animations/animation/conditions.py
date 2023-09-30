from typing import Iterable

from .condition import Condition


class All(Condition):
    __slots__ = ("_conditions",)

    def __init__(
        self,
        conditions: Iterable[Condition]
    ) -> None:
        super().__init__()
        self._conditions: tuple[Condition, ...] = tuple(conditions)

    def judge(self) -> bool:
        return all(condition.judge() for condition in self._conditions)


class Any(Condition):
    __slots__ = ("_conditions",)

    def __init__(
        self,
        conditions: Iterable[Condition]
    ) -> None:
        super().__init__()
        self._conditions: tuple[Condition, ...] = tuple(conditions)

    def judge(self) -> bool:
        return any(condition.judge() for condition in self._conditions)


class Always(Condition):
    __slots__ = ()

    def judge(self) -> bool:
        return True


class Never(Condition):
    __slots__ = ()

    def judge(self) -> bool:
        return False


class Conditions:
    __slots__ = ()

    def __new__(cls):
        raise TypeError

    @classmethod
    def all(
        cls,
        conditions: Iterable[Condition]
    ) -> All:
        return All(conditions)

    @classmethod
    def any(
        cls,
        conditions: Iterable[Condition]
    ) -> Any:
        return Any(conditions)

    @classmethod
    def always(cls) -> Always:
        return Always()

    @classmethod
    def never(cls) -> Never:
        return Never()

    #@classmethod
    #def launched(
    #    cls,
    #    animation: "Animation"
    #) -> Launched:
    #    return Launched(animation)

    #@classmethod
    #def terminated(
    #    cls,
    #    animation: "Animation"
    #) -> Terminated:
    #    return Terminated(animation)

    #@classmethod
    #def progress_duration(
    #    cls,
    #    animation: "Animation",
    #    delta_alpha: float
    #) -> ProgressDuration:
    #    return ProgressDuration(animation, delta_alpha)

    #@classmethod
    #def event_captured(
    #    cls,
    #    event: Event
    #) -> EventCaptured:
    #    return EventCaptured(event)
