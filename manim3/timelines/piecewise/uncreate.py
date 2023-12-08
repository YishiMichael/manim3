from __future__ import annotations


from typing import Self

from ...mobjects.mobject import Mobject
from ..timeline.timeline import Timeline
from .create import PartialPiecewiser


class Uncreate(Timeline):
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
        self._piecewiser: PartialPiecewiser = PartialPiecewiser(
            n_segments=n_segments,
            backwards=backwards
        )

    async def construct(
        self: Self
    ) -> None:
        mobject = self._mobject

        await self.play(mobject.animate(rewind=True).piecewise(mobject.copy(), self._piecewiser))
        self.scene.discard(mobject)
