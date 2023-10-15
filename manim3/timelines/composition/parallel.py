from __future__ import annotations


from typing import Self

from ..timeline.conditions import Conditions
from ..timeline.rate import Rate
from ..timeline.timeline import Timeline
from .lagged import Lagged


class Parallel(Timeline):
    __slots__ = (
        "_timeline_items",
        "_rate"
    )

    def __init__(
        self: Self,
        *timelines: Timeline,
        rate: Rate | None = None,
        lag_time: float = 0.0,
        lag_ratio: float = 0.0
    ) -> None:
        accumulated_lag_time = 0.0
        timeline_items: list[tuple[Timeline, float]] = []
        for timeline in timelines:
            timeline_items.append((timeline, accumulated_lag_time))
            accumulated_lag_time += lag_time + lag_ratio * timeline._run_alpha
        super().__init__(
            run_alpha=max((
                timeline_lag_time + timeline._run_alpha
                for timeline, timeline_lag_time in timeline_items
            ), default=0.0)
        )
        self._timeline_items: tuple[tuple[Timeline, float], ...] = tuple(timeline_items)
        self._rate: Rate | None = rate

    async def construct(
        self: Self
    ) -> None:
        timeline_items = self._timeline_items
        rate = self._rate
        for timeline, timeline_lag_time in timeline_items:
            self.prepare(Lagged(timeline, lag_time=timeline_lag_time), rate=rate)
        await self.wait_until(Conditions.all(
            timeline.terminated() for timeline, _ in timeline_items
        ))
