from typing import (
    Callable,
    TypeVar
)

from ...mobjects.mobject.mobject import Mobject
from ..transform.transform_to_copy import TransformToCopy


_MobjectT = TypeVar("_MobjectT", bound=Mobject)


class FadeOut(TransformToCopy):
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
        await super().timeline()
        self.scene.discard(self._start_mobject)
