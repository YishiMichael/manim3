from __future__ import annotations


from typing import Self

from ...animatables.animatable.piecewiser import Piecewiser
from ...animatables.animatable.piecewisers import Piecewisers
from ...mobjects.mobject import Mobject
from ..timeline.timeline import Timeline


class Create(Timeline):
    __slots__ = (
        "_mobject",
        "_piecewiser"
    )

    def __init__(
        self: Self,
        mobject: Mobject,
        *,
        n_segments: int = 1,
        backwards: bool = False
    ) -> None:
        super().__init__(run_alpha=1.0)
        self._mobject: Mobject = mobject
        self._piecewiser: Piecewiser = Piecewisers.partial(
            n_segments=n_segments,
            backwards=backwards
        )

    async def construct(
        self: Self
    ) -> None:
        mobject = self._mobject

        self.scene.add(mobject)
        await self.play(mobject.animate().piecewise(mobject.copy(), self._piecewiser))
