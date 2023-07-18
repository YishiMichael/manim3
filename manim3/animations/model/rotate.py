from ...constants.custom_typing import NP_3f8
from ...mobjects.mobject.abouts.about import About
from ...mobjects.mobject.mobject import Mobject
from ...mobjects.mobject.model_interpolants.rotate_model_interpolant import RotateModelInterpolant
from .model_finite_base import ModelFiniteBase


class Rotate(ModelFiniteBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        rotvec: NP_3f8,
        about: About | None = None,
        *,
        arrive: bool = False
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=RotateModelInterpolant(rotvec),
            about=about,
            arrive=arrive
        )
