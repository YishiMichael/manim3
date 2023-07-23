from ....constants.custom_typing import NP_3f8
from ..mobject import Mobject
from .about import About


class AboutPosition(About):
    __slots__ = ("_position",)

    def __init__(
        self,
        position: NP_3f8
    ) -> None:
        super().__init__()
        self._position: NP_3f8 = position

    def _get_about_position(
        self,
        mobject: Mobject
    ) -> NP_3f8:
        return self._position
