from ...constants.custom_typing import NP_3f8
from ...mobjects.mobject.mobject import Mobject
from ...mobjects.mobject.model_interpolants.shift_model_interpolant import ShiftModelInterpolant
from .model_finite_base import ModelFiniteBase


class Shift(ModelFiniteBase):
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
            model_interpolant=ShiftModelInterpolant(vector),
            arrive=arrive
        )
