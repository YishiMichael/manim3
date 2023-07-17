from typing import Callable


class RateUtils:
    __slots__ = ()

    def __new__(cls):
        raise TypeError

    @classmethod
    def compose_rates(
        cls,
        rate_0: Callable[[float], float],
        rate_1: Callable[[float], float]
    ) -> Callable[[float], float]:

        def result(
            x: float
        ) -> float:
            return rate_0(rate_1(x))

        return result

    @classmethod
    def lag_rate(
        cls,
        rate: Callable[[float], float],
        lag_time: float
    ) -> Callable[[float], float]:

        def result(
            t: float
        ) -> float:
            return rate(t - lag_time)

        return result

    @classmethod
    def scale_rate(
        cls,
        rate: Callable[[float], float],
        run_time_scale: float,
        run_alpha_scale: float
    ) -> Callable[[float], float]:

        def result(
            t: float
        ) -> float:
            return rate(t / run_time_scale) * run_alpha_scale

        return result

    # Rate functions, defined on `[0, 1] -> [0, 1]`.

    @classmethod
    def linear(
        cls,
        t: float
    ) -> float:
        return t

    @classmethod
    def smooth(
        cls,
        t: float
    ) -> float:
        # Zero first and second derivatives at `t=0` and `t=1`.
        # Equivalent to `bezier([0, 0, 0, 1, 1, 1])`.
        s = 1.0 - t
        return (t ** 3) * (10.0 * s * s + 5.0 * s * t + t * t)

    @classmethod
    def rush_into(
        cls,
        t: float
    ) -> float:
        return 2.0 * cls.smooth(0.5 * t)

    @classmethod
    def rush_from(
        cls,
        t: float
    ) -> float:
        return 2.0 * cls.smooth(0.5 * (t + 1.0)) - 1.0
