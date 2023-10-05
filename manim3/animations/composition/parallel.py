from __future__ import annotations


from typing import Self

from ..animation.animation import Animation
from ..animation.conditions import Conditions
from ..animation.rate import Rate
from .lagged import Lagged


class Parallel(Animation):
    __slots__ = (
        "_animation_items",
        "_rate"
    )

    def __init__(
        self: Self,
        *animations: Animation,
        rate: Rate | None = None,
        lag_time: float = 0.0,
        lag_ratio: float = 0.0
    ) -> None:
        accumulated_lag_time = 0.0
        animation_items: list[tuple[Animation, float]] = []
        for animation in animations:
            animation_items.append((animation, accumulated_lag_time))
            accumulated_lag_time += lag_time + lag_ratio * animation._run_alpha
        super().__init__(
            run_alpha=max((
                animation_lag_time + animation._run_alpha
                for animation, animation_lag_time in animation_items
            ), default=0.0)
        )
        self._animation_items: list[tuple[Animation, float]] = animation_items
        self._rate: Rate | None = rate

    async def timeline(
        self: Self
    ) -> None:
        animation_items = self._animation_items
        rate = self._rate
        for animation, animation_lag_time in animation_items:
            self.prepare(Lagged(animation, lag_time=animation_lag_time), rate=rate)
        await self.wait_until(Conditions.all(
            animation.terminated() for animation, _ in animation_items
        ))
