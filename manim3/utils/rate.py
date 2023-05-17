from typing import Callable

import numpy as np
import scipy.optimize


class RateUtils:
    __slots__ = ()

    def __new__(cls):
        raise TypeError

    @classmethod
    def compose(
        cls,
        *funcs: Callable[[float], float]
    ) -> Callable[[float], float]:

        def result(
            x: float
        ) -> float:
            y = x
            for func in reversed(funcs):
                y = func(y)
            return y

        return result

    @classmethod
    def inverse(
        cls,
        func: Callable[[float], float]
    ) -> Callable[[float], float]:

        def result(
            y: float
        ) -> float:
            for x0 in np.linspace(0.0, 1.0, 5):
                optimize_result = scipy.optimize.root(lambda x: func(x) - y, x0)
                if optimize_result.success:
                    return float(optimize_result.x)
            raise ValueError

        return result

    @classmethod
    def adjust(
        cls,
        func: Callable[[float], float],
        *,
        lag_time: float = 0.0,
        run_time_scale: float = 1.0,
        run_alpha_scale: float = 1.0
    ) -> Callable[[float], float]:

        def result(
            t: float
        ) -> float:
            return func((t - lag_time) / run_time_scale) * run_alpha_scale

        return result

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
