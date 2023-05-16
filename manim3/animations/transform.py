from typing import Callable

from ..animations.animation import Animation
from ..mobjects.mobject import (
    Mobject,
    MobjectMeta
)
from ..utils.rate import RateUtils


class Transform(Animation):
    __slots__ = (
        "_start_mobject",
        "_stop_mobject",
        "_intermediate_mobject"
    )

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        intermediate_mobject: Mobject = Mobject()
        callbacks: list[Callable[[float], None]] = []
        # Requires descendants of `start_mobject` and `stop_mobject` perfectly aligned.
        for descendant_0, descendant_1 in zip(
            start_mobject.iter_descendants(),
            stop_mobject.iter_descendants(),
            strict=True
        ):
            intermediate_descendant = descendant_0.copy_standalone()
            intermediate_mobject.add(intermediate_descendant)
            callbacks.append(MobjectMeta._interpolate(descendant_0, descendant_1)(intermediate_descendant))

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

    async def timeline(self) -> None:
        start_mobject = self._start_mobject
        stop_mobject = self._stop_mobject
        intermediate_mobject = self._intermediate_mobject
        parents = list(start_mobject.iter_parents())
        start_mobject.discarded_by(*parents)
        intermediate_mobject.added_by(*parents)
        await self.wait()
        intermediate_mobject.discarded_by(*parents)
        stop_mobject.added_by(*parents)
