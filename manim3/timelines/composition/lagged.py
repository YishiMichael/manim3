from __future__ import annotations


from typing import Self

from ..timeline.timeline import Timeline
from ..timeline.timeline_protocol import TimelineProtocol
from ..timeline.rate import Rate


class Lagged(Timeline):
    __slots__ = (
        "_timeline",
        "_rate",
        "_lag_time"
    )

    def __init__(
        self: Self,
        supports_timeline: TimelineProtocol,
        *,
        rate: Rate | None = None,
        lag_time: float = 0.0
    ) -> None:
        timeline = supports_timeline._submit_timeline()
        super().__init__(
            run_alpha=lag_time + timeline._run_alpha
        )
        self._timeline: Timeline = timeline
        self._rate: Rate | None = rate
        self._lag_time: float = lag_time

    async def construct(
        self: Self
    ) -> None:
        await self.wait(self._lag_time)
        await self.play(self._timeline, rate=self._rate)
