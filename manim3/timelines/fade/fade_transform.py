from __future__ import annotations


from typing import Self

from ...mobjects.mobject import Mobject
from ..composition.parallel import Parallel
from ..timeline import Timeline
from .fade_in import FadeIn
from .fade_out import FadeOut


class FadeTransform(Timeline):
    __slots__ = (
        "_start_mobject",
        "_stop_mobject"
    )

    def __init__(
        self: Self,
        start_mobject: Mobject,
        stop_mobject: Mobject
    ) -> None:
        super().__init__(run_alpha=1.0)
        self._start_mobject: Mobject = start_mobject
        self._stop_mobject: Mobject = stop_mobject

    async def construct(
        self: Self
    ) -> None:
        start_mobject = self._start_mobject
        stop_mobject = self._stop_mobject

        self.scene.add(stop_mobject)
        await self.play(Parallel(
            FadeOut(start_mobject),
            start_mobject.animate().move_to(stop_mobject),
            FadeIn(stop_mobject),
            stop_mobject.animate(rewind=True).move_to(start_mobject)
        ))
        self.scene.discard(start_mobject)
