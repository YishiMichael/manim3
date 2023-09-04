from ...mobjects.mobject.abouts.about import About
from ...mobjects.mobject.remodel_handlers.remodel_handler import RemodelHandler
from ...mobjects.mobject.mobject import (
    Mobject,
    RemodelBoundHandler
)
from ..animation.animation import Animation


class RemodelBase(Animation):
    __slots__ = (
        "_mobject_to_original_model_matrix",
        "_remodel_handler",
        "_about"
    )

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
        self._remodel_bound_handlers: list[RemodelBoundHandler] = mobject._get_remodel_bound_handlers(
            remodel_handler=remodel_handler,
            about=about
        )
        #self._mobject: Mobject = mobject
        #self._original_model_matrix_dict: dict[Mobject, NP_44f8] = {
        #    descendant: descendant._model_matrix_._array_
        #    for descendant in mobject.iter_descendants()
        #}
        #self._remodel_handler: RemodelHandler = remodel_handler
        #self._about: About | None = about

    def updater(
        self,
        alpha: float
    ) -> None:
        #remodel_handler = self._remodel_handler
        #about = self._about
        #for mobject, original_model_matrix in self._original_model_matrix_dict.items():
        #    mobject._remodel(
        #        remodel_matrix=remodel_handler._remodel(alpha),
        #        about=about,
        #        original_model_matrix=original_model_matrix
        #    )
        for remodel_bound_handler in self._remodel_bound_handlers:
            remodel_bound_handler._remodel(alpha)
