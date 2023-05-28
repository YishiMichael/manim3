from typing import (
    Callable,
    TypeVar
)

from ..mobjects.mobject import Mobject
from ..utils.rate import RateUtils
from .composition import Parallel
from .transform import (
    TransformFromCopy,
    TransformToCopy
)


_MobjectT = TypeVar("_MobjectT", bound=Mobject)


class FadeIn(TransformFromCopy):
    __slots__ = ()

    def __init__(
        self,
        mobject: _MobjectT,
        func: Callable[[_MobjectT], _MobjectT] = lambda mob: mob,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            mobject=mobject,
            func=lambda mob: func(mob.set_style(opacity=0.0)),
            run_time=run_time,
            rate_func=rate_func
        )

    async def timeline(self) -> None:
        self.scene.add(self._stop_mobject)
        await super().timeline()


class FadeOut(TransformToCopy):
    __slots__ = ()

    def __init__(
        self,
        mobject: _MobjectT,
        func: Callable[[_MobjectT], _MobjectT] = lambda mob: mob,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            mobject=mobject,
            func=lambda mob: func(mob.set_style(opacity=0.0)),
            run_time=run_time,
            rate_func=rate_func
        )

    async def timeline(self) -> None:
        await super().timeline()
        self.scene.discard(self._start_mobject)


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
            FadeOut(
                mobject=intermediate_start_mobject,
                func=lambda mob: mob.match_bounding_box(stop_mobject)
            ),
            FadeIn(
                mobject=intermediate_stop_mobject,
                func=lambda mob: mob.match_bounding_box(start_mobject)
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
        self.scene.discard(self._start_mobject)
        self.scene.add(self._intermediate_mobject)
        await super().timeline()
        self.scene.discard(self._intermediate_mobject)
        self.scene.add(self._stop_mobject)
