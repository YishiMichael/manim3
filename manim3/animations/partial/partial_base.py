from typing import Callable

from ...mobjects.mobject.operation_handlers.partial_bound_handler import PartialBoundHandler
from ...mobjects.mobject.mobject import Mobject
from ..animation.animation import Animation


class PartialBase(Animation):
    __slots__ = (
        "_partial_bound_handlers",
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
        self._partial_bound_handlers: list[PartialBoundHandler] = [
            PartialBoundHandler(descendant, descendant)
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
        for partial_bound_handler in self._partial_bound_handlers:
            partial_bound_handler.partial(alpha_0, alpha_1)

    async def timeline(self) -> None:
        await self.wait()
