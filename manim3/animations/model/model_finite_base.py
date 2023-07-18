from ...mobjects.mobject.abouts.about import About
from ...mobjects.mobject.mobject import Mobject
from ...mobjects.mobject.model_interpolants.model_interpolant import ModelInterpolant
from .model_base import ModelBase


class ModelFiniteBase(ModelBase):
    __slots__ = ("_arrive",)

    def __init__(
        self,
        mobject: Mobject,
        model_interpolant: ModelInterpolant,
        about: About | None = None,
        *,
        arrive: bool = False
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=model_interpolant,
            about=about,
            run_alpha=1.0
        )
        self._arrive: bool = arrive

    def updater(
        self,
        alpha: float
    ) -> None:
        alpha_lagging = 1.0 if self._arrive else 0.0
        super().updater(alpha - alpha_lagging)

    async def timeline(self) -> None:
        await self.wait()
