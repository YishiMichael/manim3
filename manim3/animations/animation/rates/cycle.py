from .linear import Linear
from .rate import Rate


class Cycle(Rate):
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
        remainder = t % 2.0
        return self._rate.at(remainder if remainder < 1.0 else 2.0 - remainder)
