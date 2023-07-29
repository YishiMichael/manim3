from .rush_from import RushFrom


class Smooth(RushFrom):
    __slots__ = ()

    def at(
        self,
        t: float
    ) -> float:
        return (super().at(2.0 * t - 1.0) + 1.0) / 2.0
