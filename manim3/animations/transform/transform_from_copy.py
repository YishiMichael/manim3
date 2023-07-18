from typing import (
    Callable,
    TypeVar
)

from ...mobjects.mobject.mobject import Mobject
from .transform_from import TransformFrom


_MobjectT = TypeVar("_MobjectT", bound=Mobject)


class TransformFromCopy(TransformFrom):
    __slots__ = ()

    def __init__(
        self,
        mobject: _MobjectT,
        func: Callable[[_MobjectT], _MobjectT]
    ) -> None:
        super().__init__(
            start_mobject=func(mobject.copy()),
            stop_mobject=mobject
        )
