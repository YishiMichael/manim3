from ....constants.custom_typing import NP_3f8
from ..abouts.about_edge import AboutEdge
from .align import Align


class AlignEdge(Align):
    __slots__ = ()

    def __init__(
        self,
        edge: NP_3f8,
        buff: float | NP_3f8 = 0.0
    ) -> None:
        super().__init__(
            about=AboutEdge(edge),
            direction=edge,
            buff=buff
        )
