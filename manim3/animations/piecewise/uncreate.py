from __future__ import annotations


from typing import Self

from ...animatables.animatable.piecewisers import Piecewisers
from ...mobjects.mobject import Mobject
from ..animation.animation import Animation


class Uncreate(Animation):
    __slots__ = (
        "_animation",
        "_mobject"
    )

    def __init__(
        self: Self,
        mobject: Mobject,
        *,
        n_segments: int = 1,
        backwards: bool = False
    ) -> None:
        super().__init__(run_alpha=1.0)
        self._animation: Animation = mobject.animate.piecewise(mobject.copy(), Piecewisers.partial(
            n_segments=n_segments,
            backwards=backwards
        )).build(rewind=True)
        self._mobject: Mobject = mobject

    async def timeline(
        self: Self
    ) -> None:
        await self.play(self._animation)
        self.scene.discard(self._mobject)
