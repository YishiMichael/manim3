from __future__ import annotations


from typing import (
    Iterable,
    Never,
    Self
)

from .condition import Condition


class AllCondition(Condition):
    __slots__ = ("_conditions",)

    def __init__(
        self: Self,
        conditions: Iterable[Condition]
    ) -> None:
        super().__init__()
        self._conditions: tuple[Condition, ...] = tuple(conditions)

    def judge(
        self: Self
    ) -> bool:
        return all(condition.judge() for condition in self._conditions)


class AnyCondition(Condition):
    __slots__ = ("_conditions",)

    def __init__(
        self: Self,
        conditions: Iterable[Condition]
    ) -> None:
        super().__init__()
        self._conditions: tuple[Condition, ...] = tuple(conditions)

    def judge(
        self: Self
    ) -> bool:
        return any(condition.judge() for condition in self._conditions)


class AlwaysCondition(Condition):
    __slots__ = ()

    def judge(
        self: Self
    ) -> bool:
        return True


class NeverCondition(Condition):
    __slots__ = ()

    def judge(
        self: Self
    ) -> bool:
        return False


class Conditions:
    __slots__ = ()

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @classmethod
    def all(
        cls: type[Self],
        conditions: Iterable[Condition]
    ) -> AllCondition:
        return AllCondition(conditions)

    @classmethod
    def any(
        cls: type[Self],
        conditions: Iterable[Condition]
    ) -> AnyCondition:
        return AnyCondition(conditions)

    @classmethod
    def always(
        cls: type[Self]
    ) -> AlwaysCondition:
        return AlwaysCondition()

    @classmethod
    def never(
        cls: type[Self]
    ) -> NeverCondition:
        return NeverCondition()

    #@classmethod
    #def launched(
    #    cls,
    #    timeline: "Timeline"
    #) -> Launched:
    #    return Launched(timeline)

    #@classmethod
    #def terminated(
    #    cls,
    #    timeline: "Timeline"
    #) -> Terminated:
    #    return Terminated(timeline)

    #@classmethod
    #def progress_duration(
    #    cls,
    #    timeline: "Timeline",
    #    delta_alpha: float
    #) -> ProgressDuration:
    #    return ProgressDuration(timeline, delta_alpha)

    #@classmethod
    #def event_captured(
    #    cls,
    #    event: Event
    #) -> EventCaptured:
    #    return EventCaptured(event)