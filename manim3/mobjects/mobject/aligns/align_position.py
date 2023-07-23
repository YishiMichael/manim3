from ....constants.constants import ORIGIN
from ....constants.custom_typing import NP_3f8
from ..abouts.about_position import AboutPosition
from .align import Align


class AlignPosition(Align):
    __slots__ = ()

    def __init__(
        self,
        position: NP_3f8,
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0
    ) -> None:
        super().__init__(
            about=AboutPosition(position),
            direction=direction,
            buff=buff
        )
        self._position: NP_3f8 = position
