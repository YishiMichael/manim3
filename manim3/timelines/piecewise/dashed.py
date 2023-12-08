from __future__ import annotations


from typing import Self

from ...animatables.animatable.piecewiser import Piecewiser
from ...mobjects.mobject import Mobject
from ..timeline.timeline import Timeline


class DashedPiecewiser(Piecewiser):
    __slots__ = ("_proportion",)

    def __init__(
        self: Self,
        proportion: float,
        n_segments: int,
        backwards: bool
    ) -> None:
        super().__init__(
            n_segments=n_segments,
            backwards=backwards
        )
        self._proportion: float = proportion

    def get_segment(
        self: Self,
        alpha: float
    ) -> tuple[float, float]:
        proportion = self._proportion
        return (alpha, alpha + proportion)


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
        self._piecewiser: DashedPiecewiser = DashedPiecewiser(
            proportion=proportion,
            n_segments=n_segments,
            backwards=backwards
        )

    async def construct(
        self: Self
    ) -> None:
        mobject = self._mobject

        await self.play(mobject.animate(infinite=True).piecewise(mobject.copy(), self._piecewiser))
