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
            #run_time=run_time,
            #relative_rate=relative_rate
        )
        self._animation: Animation = animation
        self._rate: Callable[[float], float] | None = rate
        self._lag_ratio: float = lag_ratio

    async def timeline(self) -> None:
        await self.wait(self._lag_ratio)
        await self.play(self._animation, rate=self._rate)


#class CompositionBase(Animation):
#    __slots__ = (
#        "_animations",
#        "_lagged_animations"
#        #"_element_rate",
#        #"_lag_ratio"
#    )

#    def __init__(
#        self,
#        *animations: Animation,
#        element_rate: Callable[[float], float] | None = None,
#        lag_ratio: float = 0.0,
#        run_alpha: float = float("inf")
#        #run_time: float | None = None,
#        #relative_rate: Callable[[float], float] = RateUtils.linear
#    ) -> None:
#        #run_alpha = sum((animation._play_run_time for animation in animations), start=0.0)
#        #if run_time is None:
#        #    run_time = RateUtils.inverse(rate_func)(run_alpha)
#        #run_alpha: float | None = 0.0
#        #for animation in animations:
#        #    if (animation_run_alpha := animation._run_alpha) is None:
#        #        run_alpha = None
#        #        break
#        #    assert run_alpha is not None
#        #    run_alpha += animation_run_alpha
#        super().__init__(
#            run_alpha=run_alpha
#            #run_alpha=sum((animation._run_alpha for animation in animations), start=0.0) + (len(animations) - 1) * lag_ratio
#            #run_time=run_time,
#            #relative_rate=relative_rate
#        )
#        self._animations: list[Animation] = list(animations)
#        self._lagged_animations: list[Animation] = [
#            Lagged(animation, rate=element_rate, lag_ratio=lag_ratio)
#            for animation in animations
#        ]
#        #self._element_rate: Callable[[float], float] | None = element_rate
#        #self._lag_ratio: float = lag_ratio


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
        #run_time: float | None = None,
        #relative_rate: Callable[[float], float] = RateUtils.linear
    ) -> None:
        #run_alpha = sum((animation._play_run_time for animation in animations), start=0.0)
        #if run_time is None:
        #    run_time = RateUtils.inverse(rate_func)(run_alpha)
        #run_alpha: float | None = 0.0
        #for animation in animations:
        #    if (animation_run_alpha := animation._run_alpha) is None:
        #        run_alpha = None
        #        break
        #    assert run_alpha is not None
        #    run_alpha += animation_run_alpha
        super().__init__(
            run_alpha=sum(
                animation._run_alpha
                for animation in animations
            ) + max(len(animations) - 1, 0) * lag_ratio
            #run_time=run_time,
            #relative_rate=relative_rate
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
        #for (animation, lagged_animation) in enumerate(zip(self._animations, self._lagged_animations, strict=True)):
        #    if not index:
        #        self.prepare(animation)
        #    #await self.wait(lag_ratio)
        #    self.prepare(animation, rate=element_rate)
        #    await self.wait_until(animation.terminated())
        #    #self.prepare(animation)
        #    #await self.wait(animation._play_run_time)


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
        #run_time: float | None = None,
        #relative_rate: Callable[[float], float] = RateUtils.linear
    ) -> None:
        #run_alpha = max((animation._play_run_time for animation in animations), default=0.0)
        #if run_time is None:
        #    run_time = RateUtils.inverse(rate_func)(run_alpha)
        super().__init__(
            run_alpha=max((
                index * lag_ratio + animation._run_alpha
                for index, animation in enumerate(animations)
            ), default=0.0)
            #run_time=run_time,
            #relative_rate=RateUtils.adjust(rate_func, run_time_scale=run_time, run_alpha_scale=run_alpha)
            #relative_rate=relative_rate,
            #terminate_condition=self.all(
            #    animation.terminated() for animation in animations
            #)
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
        #animations = self._animations
        #element_rate = self._element_rate
        #lag_ratio = self._lag_ratio
        #for animation in animations:
        #    await self.wait(lag_ratio)
        #    self.prepare(animation, rate=element_rate)
        #await self.wait_until(self.all(
        #    animation.terminated() for animation in animations
        #))
        #await self.wait(self._relative_rate(self._play_run_time))


#class Wait(Animation):
#    __slots__ = ()

#    def __init__(
#        self,
#        run_time: float = 1.0
#    ) -> None:
#        super().__init__(
#            run_time=run_time
#        )

#    async def timeline(self) -> None:
#        await self.wait(self._play_run_time)


#class Lagged(Series):
#    def __init__(
#        self,
#        animation: Animation,
#        *,
#        lag_time: float = 1.0
#    ) -> None:
#        super().__init__(
#            Wait(lag_time),
#            animation
#        )


#class LaggedParallel(Parallel):
#    __slots__ = ()

#    def __init__(
#        self,
#        *animations: Animation,
#        lag_time: float = 1.0,
#        run_time: float | None = None,
#        rate_func: Callable[[float], float] = RateUtils.linear
#    ) -> None:
#        super().__init__(*(
#            Lagged(animation, lag_time=index * lag_time)
#            for index, animation in enumerate(animations)
#        ), run_time=run_time, rate_func=rate_func)
