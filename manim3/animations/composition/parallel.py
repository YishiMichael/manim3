import itertools as it
from typing import Callable

from ..animation.animation import Animation
from ..animation.conditions.all import All
from ..animation.conditions.launched import Launched
from ..animation.conditions.terminated import Terminated
from .lagged import Lagged


class Parallel(Animation):
    __slots__ = (
        "_animations",
        "_rate",
        "_lag_ratio"
    )

    def __init__(
        self,
        *animations: Animation,
        rate: Callable[[float], float] | None = None,
        lag_ratio: float = 0.0
    ) -> None:
        super().__init__(
            run_alpha=max((
                index * lag_ratio + animation._run_alpha
                for index, animation in enumerate(animations)
            ), default=0.0)
        )
        self._animations: list[Animation] = list(animations)
        self._rate: Callable[[float], float] | None = rate
        self._lag_ratio: float = lag_ratio

    async def timeline(self) -> None:
        animations = self._animations
        rate = self._rate
        lag_ratio = self._lag_ratio
        if animations:
            self.prepare(animations[0], rate=rate)
        for prev_animation, animation in it.pairwise(animations):
            self.prepare(
                Lagged(animation, rate=rate, lag_ratio=lag_ratio),
                launch_condition=Launched(prev_animation)
            )
        await self.wait_until(All(
            Terminated(animation) for animation in animations
        ))
