from ..animation.conditions.condition_all import ConditionAll
from ..animation.conditions.terminated import Terminated
from ..animation.rates.rate import Rate
from ..animation.animation import Animation
from .lagged import Lagged


class Parallel(Animation):
    __slots__ = (
        "_animation_items",
        "_rate"
    )

    def __init__(
        self,
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

    async def timeline(self) -> None:
        animation_items = self._animation_items
        rate = self._rate
        for animation, animation_lag_time in animation_items:
            self.prepare(Lagged(animation, lag_time=animation_lag_time), rate=rate)
        await self.wait_until(ConditionAll(
            Terminated(animation) for animation, _ in animation_items
        ))
