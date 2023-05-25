from typing import Callable

from ..animations.composition import Parallel
from ..animations.transform import (
    TransformFrom,
    TransformTo
)
from ..mobjects.mobject import Mobject
from ..utils.rate import RateUtils


class FadeIn(TransformFrom):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        bounding_box_mobject: Mobject | None = None,  # TODO
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        start_mobject = mobject.copy().set_style(opacity=0.0)
        if bounding_box_mobject is not None:
            start_mobject.match_bounding_box(bounding_box_mobject)
        super().__init__(
            start_mobject=start_mobject,
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
        bounding_box_mobject: Mobject | None = None,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        stop_mobject = mobject.copy().set_style(opacity=0.0)
        if bounding_box_mobject is not None:
            stop_mobject.match_bounding_box(bounding_box_mobject)
        super().__init__(
            start_mobject=mobject,
            stop_mobject=stop_mobject,
            run_time=run_time,
            rate_func=rate_func
        )

    async def timeline(self) -> None:
        await super().timeline()
        self.discard_from_scene(self._start_mobject)


class FadeTransform(Parallel):
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
            FadeOut(start_mobject, bounding_box_mobject=stop_mobject),
            FadeIn(stop_mobject, bounding_box_mobject=start_mobject),
            run_time=run_time,
            rate_func=rate_func
        )
