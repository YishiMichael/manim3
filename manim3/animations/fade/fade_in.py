from typing import (
    Callable,
    TypeVar
)

from ...mobjects.mobject.mobject import Mobject
from ..transform.transform_from_copy import TransformFromCopy


_MobjectT = TypeVar("_MobjectT", bound=Mobject)


class FadeIn(TransformFromCopy):
    __slots__ = ()

    def __init__(
        self,
        mobject: _MobjectT,
        func: Callable[[_MobjectT], _MobjectT] = lambda mob: mob
    ) -> None:
        super().__init__(
            mobject=mobject,
            func=lambda mob: func(mob.set_style(opacity=0.0))
        )

    async def timeline(self) -> None:
        self.scene.add(self._stop_mobject)
        await super().timeline()
