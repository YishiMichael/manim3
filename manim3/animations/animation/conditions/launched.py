from typing import TYPE_CHECKING
import weakref

from ..animation_state import AnimationState
from .condition import Condition

if TYPE_CHECKING:
    from ..animation import Animation


class Launched(Condition):
    __slots__ = ("_animation_ref",)

    def __init__(
        self,
        animation: "Animation"
    ) -> None:
        super().__init__()
        self._animation_ref: weakref.ref[Animation] = weakref.ref(animation)

    def _judge(self) -> bool:
        animation = self._animation_ref()
        return animation is None or \
            animation._animation_state in (AnimationState.ON_ANIMATION, AnimationState.AFTER_ANIMATION)
