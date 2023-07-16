from typing import Callable

import numpy as np

from ..mobjects.mobject import Mobject
from ..mobjects.mobject_style_meta import MobjectStyleMeta
#from ..utils.rate import RateUtils
from .animation import Animation


class PartialABC(Animation):
    __slots__ = ("_mobject",)

    def __init__(
        self,
        mobject: Mobject,
        alpha_to_boundary_values: Callable[[float], tuple[float, float]],
        *,
        backwards: bool = False
        #run_time: float = 1.0,
        #rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        callbacks = tuple(
            MobjectStyleMeta._partial(descendant)(descendant)
            for descendant in mobject.iter_descendants()
        )

        def updater(
            alpha: float
        ) -> None:
            start, stop = alpha_to_boundary_values(alpha)
            if backwards:
                start, stop = 1.0 - stop, 1.0 - start
            for callback in callbacks:
                callback(start, stop)

        super().__init__(
            #run_time=run_time,
            #relative_rate=RateUtils.adjust(rate_func, run_time_scale=run_time),
            updater=updater
        )
        self._mobject: Mobject = mobject


class PartialCreate(PartialABC):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        backwards: bool = False
        #run_time: float = 1.0,
        #rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:

        def alpha_to_boundary_values(
            alpha: float
        ) -> tuple[float, float]:
            return (0.0, alpha)

        super().__init__(
            mobject=mobject,
            alpha_to_boundary_values=alpha_to_boundary_values,
            backwards=backwards
            #run_time=run_time,
            #rate_func=rate_func
        )

    async def timeline(self) -> None:
        self.scene.add(self._mobject)
        await self.wait()


class PartialUncreate(PartialABC):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        backwards: bool = False
        #run_time: float = 1.0,
        #rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:

        def alpha_to_boundary_values(
            alpha: float
        ) -> tuple[float, float]:
            return (0.0, 1.0 - alpha)

        super().__init__(
            mobject=mobject,
            alpha_to_boundary_values=alpha_to_boundary_values,
            backwards=backwards
            #run_time=run_time,
            #rate_func=rate_func
        )

    async def timeline(self) -> None:
        await self.wait()
        self.scene.discard(self._mobject)


class PartialFlash(PartialABC):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        flash_proportion: float = 1.0 / 16,
        backwards: bool = False
        #run_time: float = 1.0,
        #rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        assert flash_proportion >= 0.0

        def clip_proportion(
            alpha: float
        ) -> float:
            return float(np.clip(alpha, 0.0, 1.0))

        def alpha_to_boundary_values(
            alpha: float
        ) -> tuple[float, float]:
            return (
                clip_proportion(alpha * (1.0 + flash_proportion) - flash_proportion),
                clip_proportion(alpha * (1.0 + flash_proportion))
            )

        super().__init__(
            mobject=mobject,
            alpha_to_boundary_values=alpha_to_boundary_values,
            backwards=backwards
            #run_time=run_time,
            #rate_func=rate_func
        )

    async def timeline(self) -> None:
        self.scene.add(self._mobject)
        await self.wait()
        self.scene.discard(self._mobject)
