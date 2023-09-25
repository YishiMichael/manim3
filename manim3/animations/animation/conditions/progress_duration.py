import weakref
from typing import TYPE_CHECKING

from ....toplevel.toplevel import Toplevel
from ..animating_states import OnAnimating
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
        assert isinstance(animating_state := animation._animating_state, OnAnimating)
        #assert animation._absolute_rate is not None
        self._animation_ref: weakref.ref[Animation] = weakref.ref(animation)
        self._target_alpha: float = animating_state.absolute_rate.at(Toplevel.scene._timestamp) + delta_alpha

    def judge(self) -> bool:
        animation = self._animation_ref()
        return (
            animation is None
            or not isinstance(animating_state := animation._animating_state, OnAnimating)
            or animating_state.absolute_rate.at(Toplevel.scene._timestamp) >= self._target_alpha
        )
