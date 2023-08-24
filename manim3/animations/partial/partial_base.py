from typing import Callable

from ...mobjects.mobject.operation_handlers.split_bound_handler import SplitBoundHandler
from ...mobjects.mobject.mobject import Mobject
from ..animation.animation import Animation


class PartialBase(Animation):
    __slots__ = (
        "_split_bound_handlers",
        "_mobject",
        "_alpha_to_boundary_values",
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
        self._split_bound_handlers: list[SplitBoundHandler] = [
            SplitBoundHandler(descendant, descendant)
            for descendant in mobject.iter_descendants()
        ]
        self._mobject: Mobject = mobject
        self._alpha_to_boundary_values: Callable[[float], tuple[float, float]] = alpha_to_boundary_values
        self._backwards: bool = backwards

    def updater(
        self,
        alpha: float
    ) -> None:
        alpha_0, alpha_1 = self._alpha_to_boundary_values(alpha)
        if self._backwards:
            alpha_0, alpha_1 = 1.0 - alpha_1, 1.0 - alpha_0
        for split_bound_handler in self._split_bound_handlers:
            split_bound_handler.split(alpha_0, alpha_1)

    async def timeline(self) -> None:
        await self.wait()
