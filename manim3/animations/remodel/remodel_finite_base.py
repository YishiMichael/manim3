from ...mobjects.mobject.abouts.about import About
from ...mobjects.mobject.remodel_handlers.remodel_handler import RemodelHandler
from ...mobjects.mobject.mobject import Mobject
from .remodel_base import RemodelBase


class RemodelFiniteBase(RemodelBase):
    __slots__ = ("_arrive",)

    def __init__(
        self,
        mobject: Mobject,
        remodel_handler: RemodelHandler,
        about: About | None = None,
        *,
        arrive: bool = False
    ) -> None:
        super().__init__(
            mobject=mobject,
            remodel_handler=remodel_handler,
            about=about,
            run_alpha=1.0
        )
        self._arrive: bool = arrive

    def update(
        self,
        alpha: float
    ) -> None:
        alpha_lagging = 1.0 if self._arrive else 0.0
        super().update(alpha - alpha_lagging)

    async def timeline(self) -> None:
        await self.wait()
