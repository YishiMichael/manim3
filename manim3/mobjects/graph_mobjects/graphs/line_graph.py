import numpy as np

from ....constants.custom_typing import NP_3f8
from .polyline_graph import PolylineGraph


class LineGraph(PolylineGraph):
    __slots__ = ()

    def __init__(
        self,
        position_0: NP_3f8,
        position_1: NP_3f8
    ) -> None:
        super().__init__(positions=np.array((position_0, position_1)))
