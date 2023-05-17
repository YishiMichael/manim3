from typing import Callable

from ..animations.transform import (
    Transform,
    TransformFrom
)
from ..mobjects.mobject import Mobject
from ..utils.rate import RateUtils


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
            start_mobject=mobject.copy().set_style(opacity=0.0, handle_related_styles=False),
            stop_mobject=mobject,
            run_time=run_time,
            rate_func=rate_func
        )

    async def timeline(self) -> None:
        self.add_to_scene(self._stop_mobject)
        await super().timeline()


class FadeOut(Transform):
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
            stop_mobject=mobject.copy().set_style(opacity=0.0, handle_related_styles=False),
            run_time=run_time,
            rate_func=rate_func
        )

    async def timeline(self) -> None:
        await super().timeline()
        self.discard_from_scene(self._start_mobject)
