from __future__ import annotations


from typing import Self

from ..timeline.rate import Rate
from ..timeline.timeline import Timeline
from .parallel import Parallel


class Series(Parallel):
    __slots__ = ()

    def __init__(
        self: Self,
        *timelines: Timeline,
        rate: Rate | None = None,
        lag_time: float = 0.0,
        lag_ratio: float = 1.0
    ) -> None:
        super().__init__(
            *timelines,
            rate=rate,
            lag_time=lag_time,
            lag_ratio=lag_ratio
        )
