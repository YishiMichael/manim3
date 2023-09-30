from ...mobjects.mobject import Mobject
from ..animation.animation import Animation


class FadeOut(Animation):
    __slots__ = ("_mobject",)

    def __init__(
        self,
        mobject: Mobject
    ) -> None:
        super().__init__()
        self._mobject: Mobject = mobject

    async def timeline(self) -> None:
        mobject = self._mobject

        await self.play(mobject.animate.set(opacity=0.0).build())
        self.scene.discard(mobject)
