from ...mobjects.mobject.abouts.about import About
from ...mobjects.mobject.mobject import Mobject
from ...mobjects.mobject.model_interpolants.model_interpolant import ModelInterpolant
from .model_base import ModelBase


class ModelInfiniteBase(ModelBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        model_interpolant: ModelInterpolant,
        about: About | None = None
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=model_interpolant,
            about=about
        )

    async def timeline(self) -> None:
        await self.wait_forever()
