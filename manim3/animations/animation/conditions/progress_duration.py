import weakref
from typing import TYPE_CHECKING

from ....toplevel.toplevel import Toplevel
from ..animation_state import AnimationState
from .condition import Condition

if TYPE_CHECKING:
    from ..animation import Animation


class ProgressDuration(Condition):
    __slots__ = (
        "_animation_ref",
        "_target_alpha"
    )

    def __init__(
        self,
        animation: "Animation",
        delta_alpha: float
    ) -> None:
        assert animation._animation_state == AnimationState.ON_ANIMATION
        assert animation._absolute_rate is not None
        self._animation_ref: weakref.ref[Animation] = weakref.ref(animation)
        self._target_alpha: float = animation._absolute_rate.at(Toplevel.scene._timestamp) + delta_alpha

    def judge(self) -> bool:
        animation = self._animation_ref()
        return animation is None or animation._absolute_rate is None or \
            animation._absolute_rate.at(Toplevel.scene._timestamp) >= self._target_alpha
