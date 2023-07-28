from .rate import Rate
from .smooth import Smooth


class RushInto(Rate):
    __slots__ = ("_smooth",)

    def __init__(self) -> None:
        super().__init__()
        self._smooth: Smooth = Smooth()

    def at(
        self,
        t: float
    ) -> float:
        return 2.0 * self._smooth.at(0.5 * t)
