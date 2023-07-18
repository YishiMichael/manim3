from ....constants.constants import ORIGIN
from ....constants.custom_typing import NP_3f8
from ..abouts.about_point import AboutPoint
from .align import Align


class AlignPoint(Align):
    __slots__ = ()

    def __init__(
        self,
        point: NP_3f8,
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0
    ) -> None:
        super().__init__(
            about=AboutPoint(self._point),
            direction=direction,
            buff=buff
        )
        self._point: NP_3f8 = point
