from .linear import Linear
from .rate import Rate


class Rewind(Rate):
    __slots__ = ("_rate",)

    def __init__(
        self,
        rate: Rate = Linear()
    ) -> None:
        super().__init__()
        self._rate: Rate = rate

    def at(
        self,
        t: float
    ) -> float:
        return 1.0 - self._rate.at(t)
