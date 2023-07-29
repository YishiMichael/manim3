from .rush_from import RushFrom


class RushInto(RushFrom):
    __slots__ = ()

    def at(
        self,
        t: float
    ) -> float:
        return super().at(t - 1.0) + 1.0
