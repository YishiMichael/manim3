from typing import Callable

from ...mobjects.mobject.mobject import Mobject
from ...mobjects.mobject.mobject_style_meta import MobjectStyleMeta
from ..animation.animation import Animation


class TransformBase(Animation):
    __slots__ = (
        "_callbacks",
        "_start_mobject",
        "_stop_mobject",
        "_intermediate_mobject"
    )

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject,
        intermediate_mobject: Mobject
    ) -> None:
        super().__init__(
            run_alpha=1.0
        )
        self._callbacks: list[Callable[[float], None]] = [
            MobjectStyleMeta._interpolate(start_descendant, stop_descendant)(intermediate_descendant)
            for start_descendant, stop_descendant, intermediate_descendant in zip(
                start_mobject.iter_descendants(),
                stop_mobject.iter_descendants(),
                intermediate_mobject.iter_descendants(),
                strict=True
            )
        ]
        self._start_mobject: Mobject = start_mobject
        self._stop_mobject: Mobject = stop_mobject
        self._intermediate_mobject: Mobject = intermediate_mobject

    def updater(
        self,
        alpha: float
    ) -> None:
        for callback in self._callbacks:
            callback(alpha)

    async def timeline(self) -> None:
        await self.wait()
