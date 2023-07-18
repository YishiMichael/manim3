from ....constants.custom_typing import NP_3f8
from ..mobject import Mobject
from .about import About


class AboutPoint(About):
    __slots__ = ("_point",)

    def __init__(
        self,
        point: NP_3f8
    ) -> None:
        super().__init__()
        self._point: NP_3f8 = point

    def _get_about_point(
        self,
        mobject: Mobject
    ) -> NP_3f8:
        return self._point
