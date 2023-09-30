from ..mobjects.mobject import Mobject
from .animation.animation import Animation


class Transform(Animation):
    __slots__ = (
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
        self._start_mobject: Mobject = start_mobject
        self._stop_mobject: Mobject = stop_mobject
        self._intermediate_mobject: Mobject = start_mobject.copy()

    async def timeline(self) -> None:
        start_mobject = self._start_mobject
        stop_mobject = self._stop_mobject
        intermediate_mobject = self._intermediate_mobject

        self.scene.discard(start_mobject)
        self.scene.add(intermediate_mobject)
        await self.play(intermediate_mobject.animate.interpolate(start_mobject, stop_mobject).build())
        self.scene.discard(intermediate_mobject)
        self.scene.add(stop_mobject)
