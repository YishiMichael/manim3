from typing import (
    Callable,
    TypeVar
)

from ..mobjects.mobject import Mobject
from ..mobjects.mobject_style_meta import MobjectStyleMeta
from ..utils.rate import RateUtils
from .animation import Animation


_MobjectT = TypeVar("_MobjectT", bound=Mobject)


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
            MobjectStyleMeta._interpolate(start_descendant, stop_descendant)(intermediate_descendant)
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


class TransformToCopy(TransformTo):
    __slots__ = ()

    def __init__(
        self,
        mobject: _MobjectT,
        func: Callable[[_MobjectT], _MobjectT],
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            start_mobject=mobject,
            stop_mobject=func(mobject.copy()),
            run_time=run_time,
            rate_func=rate_func
        )


class TransformFromCopy(TransformFrom):
    __slots__ = ()

    def __init__(
        self,
        mobject: _MobjectT,
        func: Callable[[_MobjectT], _MobjectT],
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            start_mobject=func(mobject.copy()),
            stop_mobject=mobject,
            run_time=run_time,
            rate_func=rate_func
        )


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
        self.scene.discard(self._start_mobject)
        self.scene.add(self._intermediate_mobject)
        await self.wait()
        self.scene.discard(self._intermediate_mobject)
        self.scene.add(self._stop_mobject)
