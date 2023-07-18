from ...constants.custom_typing import NP_3f8
from ...mobjects.mobject.mobject import Mobject
from ...mobjects.mobject.model_interpolants.shift_model_interpolant import ShiftModelInterpolant
from .model_infinite_base import ModelInfiniteBase


class Shifting(ModelInfiniteBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        vector: NP_3f8
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ShiftModelInterpolant(vector)
        )
