from ..animation.animation import Animation
from ..animation.rate import Rate


class Lagged(Animation):
    __slots__ = (
        "_animation",
        "_rate",
        "_lag_time"
    )

    def __init__(
        self,
        animation: Animation,
        *,
        rate: Rate | None = None,
        lag_time: float = 0.0
    ) -> None:
        super().__init__(
            run_alpha=lag_time + animation._run_alpha
        )
        self._animation: Animation = animation
        self._rate: Rate | None = rate
        self._lag_time: float = lag_time

    async def timeline(self) -> None:
        await self.wait(self._lag_time)
        await self.play(self._animation, rate=self._rate)
