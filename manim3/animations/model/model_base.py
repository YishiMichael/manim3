from typing import Callable

from ...mobjects.mobject.abouts.about import About
from ...mobjects.mobject.mobject import Mobject
from ...mobjects.mobject.model_interpolants.model_interpolant import ModelInterpolant
from ..animation.animation import Animation


class ModelBase(Animation):
    __slots__ = ("_mobject_model_callback",)

    def __init__(
        self,
        mobject: Mobject,
        model_interpolant: ModelInterpolant,
        about: About | None = None,
        run_alpha: float = float("inf")
    ) -> None:
        super().__init__(
            run_alpha=run_alpha
        )
        self._mobject_model_callback: Callable[[float], None] = mobject._apply_transform_callback(model_interpolant, about)

    def updater(
        self,
        alpha: float
    ) -> None:
        self._mobject_model_callback(alpha)
