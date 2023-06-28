from ...constants import ORIGIN
from ...custom_typing import NP_3f8
from ..mobject import (
    AlignABC,
    Mobject
)
from .about_relatives import (
    AboutBorder,
    AboutEdge,
    AboutPoint
)


class AlignPoint(AlignABC):
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


class AlignMobject(AlignPoint):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0
    ) -> None:
        super().__init__(
            point=mobject.get_bounding_box_point(direction),
            direction=direction,
            buff=buff
        )


class AlignEdge(AlignABC):
    __slots__ = ()

    def __init__(
        self,
        edge: NP_3f8,
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0
    ) -> None:
        super().__init__(
            about=AboutEdge(edge),
            direction=direction,
            buff=buff
        )


class AlignBorder(AlignABC):
    __slots__ = ()

    def __init__(
        self,
        border: NP_3f8,
        direction: NP_3f8 = ORIGIN,
        buff: float | NP_3f8 = 0.0
    ) -> None:
        super().__init__(
            about=AboutBorder(border),
            direction=direction,
            buff=buff
        )
