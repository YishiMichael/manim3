from typing import TYPE_CHECKING

import numpy as np

from ....constants.custom_typing import (
    NP_3f8,
    NP_44f8
)
from ..remodel_handlers.remodel_handler import RemodelHandler
from ..remodel_handlers.shift_remodel_handler import ShiftRemodelHandler

if TYPE_CHECKING:
    from ..abouts.about import About
    from ..mobject import Mobject


class RemodelBoundHandler:
    __slots__ = (
        "_remodel_handler",
        "_post_remodel",
        "_pre_remodel",
        "_mobject_to_model_matrix"
    )

    def __init__(
        self,
        mobject: "Mobject",
        remodel_handler: RemodelHandler,
        about: "About | None" = None
    ) -> None:
        super().__init__()
        if about is None:
            post_remodel = np.identity(4)
            pre_remodel = np.identity(4)
        else:
            about_position = about._get_about_position(mobject=mobject)
            post_remodel = ShiftRemodelHandler(about_position).remodel()
            pre_remodel = ShiftRemodelHandler(-about_position).remodel()

        self._remodel_handler: RemodelHandler = remodel_handler
        self._post_remodel: NP_44f8 = post_remodel
        self._pre_remodel: NP_44f8 = pre_remodel
        self._mobject_to_model_matrix: "dict[Mobject, NP_44f8]" = {
            descendant: descendant._model_matrix_
            for descendant in mobject.iter_descendants()
        }

    def remodel(
        self,
        alpha: float | NP_3f8 = 1.0
    ) -> None:
        matrix = self._post_remodel @ self._remodel_handler.remodel(alpha) @ self._pre_remodel
        for mobject, model_matrix in self._mobject_to_model_matrix.items():
            mobject._model_matrix_ = matrix @ model_matrix
