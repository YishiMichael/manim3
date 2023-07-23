from ...mobjects.mobject.abouts.about import About
from ...mobjects.mobject.remodel_handlers.remodel_bound_handler import RemodelBoundHandler
from ...mobjects.mobject.remodel_handlers.remodel_handler import RemodelHandler
from ...mobjects.mobject.mobject import Mobject
from ..animation.animation import Animation


class RemodelBase(Animation):
    __slots__ = ("_remodel_bound_handler",)

    def __init__(
        self,
        mobject: Mobject,
        remodel_handler: RemodelHandler,
        about: About | None = None,
        run_alpha: float = float("inf")
    ) -> None:
        super().__init__(
            run_alpha=run_alpha
        )
        self._remodel_bound_handler: RemodelBoundHandler = RemodelBoundHandler(mobject, remodel_handler, about)

    def updater(
        self,
        alpha: float
    ) -> None:
        self._remodel_bound_handler.remodel(alpha)
