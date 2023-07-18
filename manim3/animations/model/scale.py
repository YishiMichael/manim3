from ...constants.custom_typing import NP_3f8
from ...mobjects.mobject.abouts.about import About
from ...mobjects.mobject.mobject import Mobject
from ...mobjects.mobject.model_interpolants.scale_model_interpolant import ScaleModelInterpolant
from .model_finite_base import ModelFiniteBase


class Scale(ModelFiniteBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        scale: float | NP_3f8,
        about: About | None = None,
        *,
        arrive: bool = False
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ScaleModelInterpolant(scale),
            about=about,
            arrive=arrive
        )
