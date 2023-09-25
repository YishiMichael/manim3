import weakref
from typing import TYPE_CHECKING

from ..animating_states import AfterAnimating
from .condition import Condition

if TYPE_CHECKING:
    from ..animation import Animation


class Terminated(Condition):
    __slots__ = ("_animation_ref",)

    def __init__(
        self,
        animation: "Animation"
    ) -> None:
        super().__init__()
        self._animation_ref: weakref.ref[Animation] = weakref.ref(animation)

    def judge(self) -> bool:
        animation = self._animation_ref()
        return animation is None or \
            isinstance(animation._animating_state, AfterAnimating)
