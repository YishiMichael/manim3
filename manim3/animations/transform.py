from ..mobjects.mobject import Mobject
from .animation.animation import Animation


class Transform(Animation):
    __slots__ = (
        "_animation",
        "_start_mobject",
        "_stop_mobject",
        "_intermediate_mobject"
    )

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject
    ) -> None:
        super().__init__(run_alpha=1.0)
        intermediate_mobject = start_mobject.copy()
        self._animation: Animation = intermediate_mobject.animate.interpolate(start_mobject, stop_mobject).build()
        self._start_mobject: Mobject = start_mobject
        self._stop_mobject: Mobject = stop_mobject
        self._intermediate_mobject: Mobject = intermediate_mobject

    async def timeline(self) -> None:
        self.scene.discard(self._start_mobject)
        self.scene.add(self._intermediate_mobject)
        await self.play(self._animation)
        self.scene.discard(self._intermediate_mobject)
        self.scene.add(self._stop_mobject)
