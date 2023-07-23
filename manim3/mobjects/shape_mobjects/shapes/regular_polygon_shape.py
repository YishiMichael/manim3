import numpy as np

from ....constants.constants import TAU
from .polygon_shape import PolygonShape


class RegularPolygonShape(PolygonShape):
    __slots__ = ()

    def __init__(
        self,
        n: int
    ) -> None:
        # By default, one of positions is at (1, 0).
        complex_positions = np.exp(1.0j * np.linspace(0.0, TAU, n, endpoint=False))
        super().__init__(positions=np.vstack((complex_positions.real, complex_positions.imag)).T)
