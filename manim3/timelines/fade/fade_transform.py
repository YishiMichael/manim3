from __future__ import annotations


from typing import Self

from ...mobjects.mobject import Mobject
from ..timeline.timeline import Timeline
from ..composition.parallel import Parallel
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
        #intermediate_start_mobject = start_mobject.copy()
        #intermediate_stop_mobject = stop_mobject.copy()
        #super().__init__(
        #    FadeOut(
        #        mobject=intermediate_start_mobject,
        #        func=lambda mob: mob.match_bounding_box(stop_mobject)
        #    ),
        #    FadeIn(
        #        mobject=intermediate_stop_mobject,
        #        func=lambda mob: mob.match_bounding_box(start_mobject)
        #    )
        #)
        super().__init__(run_alpha=1.0)
        #self._timeline: Timeline = 
        self._start_mobject: Mobject = start_mobject
        self._stop_mobject: Mobject = stop_mobject
        #self._intermediate_start_mobjects: Mobject = Mobject().add(
        #    intermediate_start_mobject,
        #    intermediate_stop_mobject
        #)
        #self._intermediate_mobject: Mobject = Mobject().add(
        #    intermediate_start_mobject,
        #    intermediate_stop_mobject
        #)

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
            intermediate_start_mobject.animate().match_box(stop_mobject),
            FadeIn(intermediate_stop_mobject),
            intermediate_stop_mobject.animate(rewind=True).match_box(start_mobject)
        ))
        self.scene.discard(intermediate_start_mobject, intermediate_stop_mobject)
        self.scene.add(stop_mobject)
