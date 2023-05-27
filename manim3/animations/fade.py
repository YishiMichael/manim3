from typing import Callable

from ..mobjects.mobject import Mobject
from ..utils.rate import RateUtils
from .composition import Parallel
from .transform import (
    TransformFrom,
    TransformTo
)


class FadeIn(TransformFrom):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            start_mobject=mobject.copy().set_style(opacity=0.0),
            stop_mobject=mobject,
            run_time=run_time,
            rate_func=rate_func
        )

    async def timeline(self) -> None:
        self.add_to_scene(self._stop_mobject)
        await super().timeline()


class FadeOut(TransformTo):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            start_mobject=mobject,
            stop_mobject=mobject.copy().set_style(opacity=0.0),
            run_time=run_time,
            rate_func=rate_func
        )

    async def timeline(self) -> None:
        await super().timeline()
        self.discard_from_scene(self._start_mobject)


class FadeTransform(Parallel):
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
        intermediate_start_mobject = start_mobject.copy().set_style(
            is_transparent=True
        )
        intermediate_stop_mobject = stop_mobject.copy().set_style(
            is_transparent=True
        )
        super().__init__(
            TransformTo(
                start_mobject=intermediate_start_mobject,
                stop_mobject=intermediate_start_mobject.copy().set_style(
                    opacity=0.0
                ).match_bounding_box(stop_mobject)
            ),
            TransformFrom(
                start_mobject=intermediate_stop_mobject.copy().set_style(
                    opacity=0.0
                ).match_bounding_box(start_mobject),
                stop_mobject=intermediate_stop_mobject
            ),
            run_time=run_time,
            rate_func=rate_func
        )
        self._start_mobject: Mobject = start_mobject
        self._stop_mobject: Mobject = stop_mobject
        self._intermediate_mobject: Mobject = Mobject().add(
            intermediate_start_mobject,
            intermediate_stop_mobject
        )

    async def timeline(self) -> None:
        self.discard_from_scene(self._start_mobject)
        self.add_to_scene(self._intermediate_mobject)
        await super().timeline()
        self.discard_from_scene(self._intermediate_mobject)
        self.add_to_scene(self._stop_mobject)
