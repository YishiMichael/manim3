from __future__ import annotations


from typing import Self

from ...animatables.animatable.piecewisers import Piecewisers
from ...mobjects.mobject import Mobject
from ..animation.animation import Animation


class Create(Animation):
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
        )).build()
        self._mobject: Mobject = mobject

    async def timeline(
        self: Self
    ) -> None:
        self.scene.add(self._mobject)
        await self.play(self._animation)
