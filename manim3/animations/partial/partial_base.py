from typing import Callable

from ...mobjects.mobject.mobject import Mobject
from ...mobjects.mobject.mobject_style_meta import MobjectStyleMeta
from ..animation.animation import Animation


class PartialBase(Animation):
    __slots__ = (
        "_mobject",
        "_alpha_to_boundary_values",
        "_callbacks",
        "_backwards"
    )

    def __init__(
        self,
        mobject: Mobject,
        alpha_to_boundary_values: Callable[[float], tuple[float, float]],
        *,
        backwards: bool = False
    ) -> None:

        super().__init__(
            run_alpha=1.0
        )
        self._mobject: Mobject = mobject
        self._alpha_to_boundary_values: Callable[[float], tuple[float, float]] = alpha_to_boundary_values
        self._callbacks: list[Callable[[float, float], None]] = [
            MobjectStyleMeta._partial(descendant)(descendant)
            for descendant in mobject.iter_descendants()
        ]
        self._backwards: bool = backwards

    def updater(
        self,
        alpha: float
    ) -> None:
        start, stop = self._alpha_to_boundary_values(alpha)
        if self._backwards:
            start, stop = 1.0 - stop, 1.0 - start
        for callback in self._callbacks:
            callback(start, stop)

    async def timeline(self) -> None:
        await self.wait()
