from typing import Callable

from ..animations.animation import Animation
from ..custom_typing import TimelineT
from ..mobjects.mobject import Mobject
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
        intermediate_mobjects_with_callback = list(
            (
                mobject_0.copy_standalone(),
                mobject_0.get_interpolate_callback(mobject_1)
            )
            for mobject_0, mobject_1 in zip(
                start_mobject.iter_descendants(),
                stop_mobject.iter_descendants(),
                strict=True
            )
        )

        def updater(
            alpha: float
        ) -> None:
            for mobject, callback in intermediate_mobjects_with_callback:
                callback(mobject, alpha)

        super().__init__(
            run_time=run_time,
            relative_rate=RateUtils.adjust(rate_func, run_time_scale=run_time),
            updater=updater
        )
        self._start_mobject: Mobject = start_mobject
        self._stop_mobject: Mobject = stop_mobject
        self._intermediate_mobject: Mobject = Mobject().add(*(
            mobject for mobject, _ in intermediate_mobjects_with_callback
        ))

    def timeline(self) -> TimelineT:
        start_mobject = self._start_mobject
        stop_mobject = self._stop_mobject
        intermediate_mobject = self._intermediate_mobject
        parents = list(start_mobject.iter_parents())
        start_mobject.discarded_by(*parents)
        intermediate_mobject.added_by(*parents)
        yield from self.wait()
        intermediate_mobject.discarded_by(*parents)
        stop_mobject.added_by(*parents)
