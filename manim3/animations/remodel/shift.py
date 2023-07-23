from ...constants.custom_typing import NP_3f8
from ...mobjects.mobject.remodel_handlers.shift_remodel_handler import ShiftRemodelHandler
from ...mobjects.mobject.mobject import Mobject
from .remodel_finite_base import RemodelFiniteBase


class Shift(RemodelFiniteBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        vector: NP_3f8,
        *,
        arrive: bool = False
    ) -> None:
        super().__init__(
            mobject=mobject,
            remodel_handler=ShiftRemodelHandler(vector),
            arrive=arrive
        )
