from __future__ import annotations


from typing import Self

from .regular_polygon import RegularPolygon


class Circle(RegularPolygon):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        super().__init__(n=64)
