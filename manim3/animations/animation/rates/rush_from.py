from .rate import Rate
from .smooth import Smooth


class RushFrom(Rate):
    __slots__ = ("_smooth",)

    def __init__(self) -> None:
        super().__init__()
        self._smooth: Smooth = Smooth()

    def at(
        self,
        t: float
    ) -> float:
        return 2.0 * self._smooth.at(0.5 * (t + 1.0)) - 1.0
