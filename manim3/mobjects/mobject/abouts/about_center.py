from ....constants.constants import ORIGIN
from .about_edge import AboutEdge


class AboutCenter(AboutEdge):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            edge=ORIGIN
        )
