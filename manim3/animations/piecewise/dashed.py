from ...animatables.animatable.piecewisers import Piecewisers
from ...mobjects.mobject import Mobject
from ..animation.animation import Animation


class Dashed(Animation):
    __slots__ = (
        "_animation",
        "_mobject"
    )

    def __init__(
        self,
        mobject: Mobject,
        *,
        proportion: float = 1.0 / 2,
        n_segments: int = 16,
        backwards: bool = False
    ) -> None:
        super().__init__(run_alpha=float("inf"))
        self._animation: Animation = mobject.animate.piecewise(mobject.copy(), Piecewisers.dashed(
            proportion=proportion,
            n_segments=n_segments,
            backwards=backwards
        )).build(infinite=True)
        self._mobject: Mobject = mobject

    async def timeline(self) -> None:
        await self.play(self._animation)
