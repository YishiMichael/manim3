from ....constants.constants import ORIGIN
from ....constants.custom_typing import NP_3f8
from ..abouts.about import About
from ..mobject import Mobject


class Align:
    __slots__ = (
        "_about",
        "_direction",
        "_buff"
    )

    def __init__(
        self,
        about: About,
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0
    ) -> None:
        super().__init__()
        self._about: About = about
        self._direction: NP_3f8 = direction
        self._buff: float | NP_3f8 = buff

    def _get_shift_vector(
        self,
        mobject: Mobject,
        direction_sign: float
    ) -> NP_3f8:
        target_position = self._about._get_about_position(mobject)
        direction = direction_sign * self._direction
        position_to_align = mobject.get_bounding_box_position(direction) + self._buff * direction
        return target_position - position_to_align
