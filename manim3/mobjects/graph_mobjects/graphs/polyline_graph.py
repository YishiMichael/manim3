import numpy as np

from ....constants.custom_typing import NP_x3f8
from .graph import Graph


class PolylineGraph(Graph):
    __slots__ = ()

    def __init__(
        self,
        positions: NP_x3f8
    ) -> None:
        assert len(positions)
        arange = np.arange(len(positions))
        super().__init__(
            positions=positions,
            edges=np.vstack((arange[:-1], arange[1:])).T
        )
