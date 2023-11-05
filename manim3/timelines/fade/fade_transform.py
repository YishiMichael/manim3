from __future__ import annotations


from typing import Self

from ...mobjects.mobject import Mobject
from ..composition.parallel import Parallel
from ..timeline.timeline import Timeline
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
        intermediate_start_mobject = start_mobject.copy()
        intermediate_stop_mobject = stop_mobject.copy()

        self.scene.discard(start_mobject)
        self.scene.add(intermediate_start_mobject, intermediate_stop_mobject)
        await self.play(Parallel(
            FadeOut(intermediate_start_mobject),
            intermediate_start_mobject.animate().move_to(stop_mobject),
            FadeIn(intermediate_stop_mobject),
            intermediate_stop_mobject.animate(rewind=True).move_to(start_mobject)
        ))
        self.scene.discard(intermediate_start_mobject, intermediate_stop_mobject)
        self.scene.add(stop_mobject)
