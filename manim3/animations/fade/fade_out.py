from ...mobjects.mobject import Mobject
from ..animation.animation import Animation


class FadeOut(Animation):
    __slots__ = (
        "_animation",
        "_mobject"
    )

    def __init__(
        self,
        mobject: Mobject
    ) -> None:
        super().__init__(run_alpha=1.0)
        self._animation: Animation = mobject.animate.set(opacity=0.0).build()
        self._mobject: Mobject = mobject

    async def timeline(self) -> None:
        await self.play(self._animation)
        self.scene.discard(self._mobject)
