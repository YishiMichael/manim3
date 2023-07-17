import itertools as it
from typing import Callable

from .animation import Animation


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


class Series(Animation):
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
            run_alpha=sum(
                animation._run_alpha
                for animation in animations
            ) + max(len(animations) - 1, 0) * lag_ratio
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
                launch_condition=prev_animation.terminated()
            )
        await self.wait_until(self.all(
            animation.terminated() for animation in animations
        ))


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
                launch_condition=prev_animation.launched()
            )
        await self.wait_until(self.all(
            animation.terminated() for animation in animations
        ))
