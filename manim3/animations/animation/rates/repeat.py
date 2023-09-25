from .linear import Linear
from .rate import Rate


class Repeat(Rate):
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
        return self._rate.at(t % 1.0)
