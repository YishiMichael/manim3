from __future__ import annotations


from typing import Self

from ...mobjects.mobject import Mobject
from ..timeline.timeline import Timeline

#from ..transform.transform_from_copy import TransformFromCopy


#_MobjectT = TypeVar("_MobjectT", bound=Mobject)


class FadeIn(Timeline):
    __slots__ = ("_mobject",)

    def __init__(
        self: Self,
        mobject: Mobject
    ) -> None:
        super().__init__(run_alpha=1.0)
        #self._timeline: Timeline = mobject.animate(rewind=True).set(opacity=0.0).submit()
        self._mobject: Mobject = mobject
        #super().__init__(
        #    mobject=mobject,
        #    func=lambda mob: func(mob.set(opacity=0.0))
        #)

    async def construct(
        self: Self
    ) -> None:
        mobject = self._mobject

        self.scene.add(mobject)
        await self.play(mobject.animate(rewind=False).set(opacity=0.0))
