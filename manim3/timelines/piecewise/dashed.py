from __future__ import annotations


from typing import Self

from ...animatables.animatable.piecewisers import (
    Piecewiser,
    Piecewisers
)
from ...mobjects.mobject import Mobject
from ..timeline.timeline import Timeline


class Dashed(Timeline):
    __slots__ = (
        "_mobject",
        "_piecewiser"
    )

    def __init__(
        self: Self,
        mobject: Mobject,
        *,
        proportion: float = 1.0 / 2,
        n_segments: int = 16,
        backwards: bool = False
    ) -> None:
        super().__init__(run_alpha=float("inf"))
        self._mobject: Mobject = mobject
        self._piecewiser: Piecewiser = Piecewisers.dashed(
            proportion=proportion,
            n_segments=n_segments,
            backwards=backwards
        )

    async def construct(
        self: Self
    ) -> None:
        mobject = self._mobject

        await self.play(mobject.animate(infinite=True).piecewise(mobject.copy(), self._piecewiser))
