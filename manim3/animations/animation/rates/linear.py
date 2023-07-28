from .rate import Rate


class Linear(Rate):
    __slots__ = ()

    def at(
        self,
        t: float
    ) -> float:
        return t
