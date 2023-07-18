from typing import Callable

from ..animation.animation import Animation


class Lagged(Animation):
    __slots__ = (
        "_animation",
        "_rate",
        "_lag_ratio"
    )

    def __init__(
        self,
        animation: Animation,
        *,
        rate: Callable[[float], float] | None = None,
        lag_ratio: float = 0.0
    ) -> None:
        super().__init__(
            run_alpha=lag_ratio + animation._run_alpha
        )
        self._animation: Animation = animation
        self._rate: Callable[[float], float] | None = rate
        self._lag_ratio: float = lag_ratio

    async def timeline(self) -> None:
        await self.wait(self._lag_ratio)
        await self.play(self._animation, rate=self._rate)
