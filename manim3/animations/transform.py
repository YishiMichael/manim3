from typing import Callable

from ..animations.animation import Animation
from ..mobjects.mobject import (
    Mobject,
    MobjectMeta
)
from ..utils.rate import RateUtils


class TransformABC(Animation):
    __slots__ = (
        "_start_mobject",
        "_stop_mobject",
        "_intermediate_mobject"
    )

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject,
        intermediate_mobject: Mobject,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        callbacks = tuple(
            MobjectMeta._interpolate(start_descendant, stop_descendant)(intermediate_descendant)
            for start_descendant, stop_descendant, intermediate_descendant in zip(
                start_mobject.iter_descendants(),
                stop_mobject.iter_descendants(),
                intermediate_mobject.iter_descendants(),
                strict=True
            )
        )

        def updater(
            alpha: float
        ) -> None:
            for callback in callbacks:
                callback(alpha)

        super().__init__(
            run_time=run_time,
            relative_rate=RateUtils.adjust(rate_func, run_time_scale=run_time),
            updater=updater
        )
        self._start_mobject: Mobject = start_mobject
        self._stop_mobject: Mobject = stop_mobject
        self._intermediate_mobject: Mobject = intermediate_mobject


class TransformTo(TransformABC):
    __slots__ = ()

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            start_mobject=start_mobject,
            stop_mobject=stop_mobject,
            intermediate_mobject=start_mobject,
            run_time=run_time,
            rate_func=rate_func
        )

    async def timeline(self) -> None:
        await self.wait()


class TransformFrom(TransformABC):
    __slots__ = ()

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            start_mobject=start_mobject,
            stop_mobject=stop_mobject,
            intermediate_mobject=stop_mobject,
            run_time=run_time,
            rate_func=rate_func
        )

    async def timeline(self) -> None:
        await self.wait()


class Transform(TransformABC):
    __slots__ = ()

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            start_mobject=start_mobject,
            stop_mobject=stop_mobject,
            intermediate_mobject=start_mobject.copy(),
            run_time=run_time,
            rate_func=rate_func
        )

    async def timeline(self) -> None:
        #start_mobject = self._start_mobject
        #stop_mobject = self._stop_mobject
        #intermediate_mobject = self._intermediate_mobject
        #parents = list(start_mobject.iter_parents())
        #start_mobject.discarded_by(*parents)
        #intermediate_mobject.added_by(*parents)
        self.discard_from_scene(self._start_mobject)
        self.add_to_scene(self._intermediate_mobject)
        await self.wait()
        self.discard_from_scene(self._intermediate_mobject)
        self.add_to_scene(self._stop_mobject)
        #intermediate_mobject.discarded_by(*parents)
        #stop_mobject.added_by(*parents)
