from ...constants.custom_typing import NP_3f8
from ...mobjects.mobject.abouts.about import About
from ...mobjects.mobject.mobject import Mobject
from ...mobjects.mobject.model_interpolants.scale_model_interpolant import ScaleModelInterpolant
from .model_infinite_base import ModelInfiniteBase


class Scaling(ModelInfiniteBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        factor: float | NP_3f8,
        about: About | None = None
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ScaleModelInterpolant(factor),
            about=about
        )
