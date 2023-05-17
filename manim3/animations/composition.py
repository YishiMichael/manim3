from typing import Callable

from ..animations.animation import Animation
from ..utils.rate import RateUtils


class Series(Animation):
    __slots__ = ("_animations",)

    def __init__(
        self,
        *animations: Animation,
        run_time: float | None = None,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        run_alpha = sum((animation._play_run_time for animation in animations), start=0.0)
        if run_time is None:
            run_time = run_alpha
        super().__init__(
            run_time=run_time,
            relative_rate=RateUtils.adjust(rate_func, run_time_scale=run_time, run_alpha_scale=run_alpha)
        )
        self._animations: tuple[Animation, ...] = animations

    async def timeline(self) -> None:
        for animation in self._animations:
            self.prepare(animation)
            await self.wait(animation._play_run_time)


class Parallel(Animation):
    __slots__ = ("_animations",)

    def __init__(
        self,
        *animations: Animation,
        run_time: float | None = None,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        run_alpha = max((animation._play_run_time for animation in animations), default=0.0)
        if run_time is None:
            run_time = run_alpha
        super().__init__(
            run_time=run_time,
            relative_rate=RateUtils.adjust(rate_func, run_time_scale=run_time, run_alpha_scale=run_alpha)
        )
        self._animations: tuple[Animation, ...] = animations

    async def timeline(self) -> None:
        self.prepare(*self._animations)
        await self.wait(self._play_run_time)


class Wait(Animation):
    __slots__ = ()

    def __init__(
        self,
        run_time: float = 1.0
    ) -> None:
        super().__init__(
            run_time=run_time
        )


class Lagged(Series):
    def __init__(
        self,
        animation: Animation,
        *,
        lag_time: float = 1.0
    ) -> None:
        super().__init__(
            Wait(lag_time),
            animation
        )


class LaggedParallel(Parallel):
    __slots__ = ()

    def __init__(
        self,
        *animations: Animation,
        lag_time: float = 1.0,
        run_time: float | None = None,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(*(
            Lagged(animation, lag_time=index * lag_time)
            for index, animation in enumerate(animations)
        ), run_time=run_time, rate_func=rate_func)
