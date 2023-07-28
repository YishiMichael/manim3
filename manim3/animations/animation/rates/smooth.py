from .rate import Rate


class Smooth(Rate):
    __slots__ = ()

    def at(
        self,
        t: float
    ) -> float:
        # Zero first and second derivatives at `t=0` and `t=1`.
        # Equivalent to `bezier([0, 0, 0, 1, 1, 1])`.
        s = 1.0 - t
        return (t ** 3) * (10.0 * s * s + 5.0 * s * t + t * t)
