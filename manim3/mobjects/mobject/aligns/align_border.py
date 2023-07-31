from ....constants.custom_typing import NP_3f8
from ..abouts.about_border import AboutBorder
from .align import Align


class AlignBorder(Align):
    __slots__ = ()

    def __init__(
        self,
        border: NP_3f8,
        buff: float | NP_3f8 = 0.0
    ) -> None:
        super().__init__(
            about=AboutBorder(border),
            direction=border,
            buff=buff
        )
