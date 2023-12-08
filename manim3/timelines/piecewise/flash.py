from __future__ import annotations


from typing import Self

from ...animatables.animatable.piecewiser import Piecewiser
from ...mobjects.mobject import Mobject
from ..timeline.timeline import Timeline


class FlashPiecewiser(Piecewiser):
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
        return (min(alpha * (1.0 + proportion), 1.0), max(alpha * (1.0 + proportion) - proportion, 0.0))


class Flash(Timeline):
    __slots__ = (
        "_mobject",
        "_piecewiser"
    )

    def __init__(
        self: Self,
        mobject: Mobject,
        *,
        proportion: float = 1.0 / 16,
        n_segments: int = 1,
        backwards: bool = False
    ) -> None:
        super().__init__(run_alpha=1.0)
        self._mobject: Mobject = mobject
        self._piecewiser: FlashPiecewiser = FlashPiecewiser(
            proportion=proportion,
            n_segments=n_segments,
            backwards=backwards
        )

    async def construct(
        self: Self
    ) -> None:
        mobject = self._mobject

        self.scene.add(mobject)
        await self.play(mobject.animate().piecewise(mobject.copy(), self._piecewiser))
        self.scene.discard(mobject)
