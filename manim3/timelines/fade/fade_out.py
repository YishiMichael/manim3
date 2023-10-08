from __future__ import annotations


from typing import Self

from ...mobjects.mobject import Mobject
from ..timeline.timeline import Timeline


class FadeOut(Timeline):
    __slots__ = ("_mobject",)

    def __init__(
        self: Self,
        mobject: Mobject
    ) -> None:
        super().__init__(run_alpha=1.0)
        self._mobject: Mobject = mobject

    async def construct(
        self: Self
    ) -> None:
        mobject = self._mobject

        await self.play(mobject.animate().set(opacity=0.0))
        self.scene.discard(mobject)
