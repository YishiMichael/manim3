from __future__ import annotations


from typing import Self

from ..timeline.rate import Rate
from ..timeline.timeline_protocol import TimelineProtocol
from .parallel import Parallel


class Series(Parallel):
    __slots__ = ()

    def __init__(
        self: Self,
        *supports_timelines: TimelineProtocol,
        rate: Rate | None = None,
        lag_time: float = 0.0,
        lag_ratio: float = 1.0
    ) -> None:
        super().__init__(
            *supports_timelines,
            rate=rate,
            lag_time=lag_time,
            lag_ratio=lag_ratio
        )