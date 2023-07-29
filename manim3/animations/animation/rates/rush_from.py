from .rate import Rate


class RushFrom(Rate):
    __slots__ = ()

    def at(
        self,
        t: float
    ) -> float:
        return (3.0 * t ** 5 - 10.0 * t ** 3 + 15.0 * t) / 8.0
