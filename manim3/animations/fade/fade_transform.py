from ...mobjects.mobject import Mobject
from ..animation.animation import Animation
from ..animation.rates import Rates
from ..composition.parallel import Parallel
from .fade_in import FadeIn
from .fade_out import FadeOut


class FadeTransform(Animation):
    __slots__ = (
        "_animation",
        "_start_mobject",
        "_stop_mobject",
        "_intermediate_start_mobjects"
    )

    def __init__(
        self,
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
        intermediate_start_mobject = start_mobject.copy()
        intermediate_stop_mobject = stop_mobject.copy()
        self._animation: Animation = Parallel(
            FadeOut(intermediate_start_mobject),
            intermediate_start_mobject.animate.match_box(stop_mobject).build(),
            FadeIn(intermediate_stop_mobject),
            intermediate_stop_mobject.animate.match_box(start_mobject).build(rate=Rates.rewind())
        )
        self._start_mobject: Mobject = start_mobject
        self._stop_mobject: Mobject = stop_mobject
        self._intermediate_start_mobjects: Mobject = Mobject().add(
            intermediate_start_mobject,
            intermediate_stop_mobject
        )
        #self._intermediate_mobject: Mobject = Mobject().add(
        #    intermediate_start_mobject,
        #    intermediate_stop_mobject
        #)

    async def timeline(self) -> None:
        #start_mobject = self._start_mobject
        #stop_mobject = self._stop_mobject
        #intermediate_start_mobject = self._intermediate_start_mobject
        #intermediate_stop_mobject = self._intermediate_stop_mobject
        self.scene.discard(self._start_mobject)
        self.scene.add(self._intermediate_start_mobjects)
        await self.play(self._animation)
        self.scene.discard(self._intermediate_start_mobjects)
        self.scene.add(self._stop_mobject)
