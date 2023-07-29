from ...mobjects.mobject.abouts.about import About
from ...mobjects.mobject.remodel_handlers.remodel_handler import RemodelHandler
from ...mobjects.mobject.mobject import Mobject
from .remodel_base import RemodelBase


class RemodelInfiniteBase(RemodelBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        remodel_handler: RemodelHandler,
        about: About | None = None
    ) -> None:
        super().__init__(
            mobject=mobject,
            remodel_handler=remodel_handler,
            about=about
        )

    async def timeline(self) -> None:
        await self.wait_forever()
