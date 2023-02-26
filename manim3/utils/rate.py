__all__ = ["RateUtils"]


from ..custom_typing import Real


class RateUtils:
    @classmethod
    def linear(
        cls,
        t: Real
    ) -> Real:
        return t

    @classmethod
    def smooth(
        cls,
        t: Real
    ) -> Real:
        # Zero first and second derivatives at t=0 and t=1.
        # Equivalent to bezier([0, 0, 0, 1, 1, 1])
        s = 1.0 - t
        return (t ** 3) * (10.0 * s * s + 5.0 * s * t + t * t)

    @classmethod
    def rush_into(
        cls,
        t: Real
    ) -> Real:
        return 2.0 * cls.smooth(0.5 * t)

    @classmethod
    def rush_from(
        cls,
        t: Real
    ) -> Real:
        return 2.0 * cls.smooth(0.5 * (t + 1.0)) - 1.0
