import numpy as np

from ....constants.custom_typing import NP_3f8
from .polyline_graph import PolylineGraph


class LineGraph(PolylineGraph):
    __slots__ = ()

    def __init__(
        self,
        start_position: NP_3f8,
        stop_position: NP_3f8
    ) -> None:
        super().__init__(positions=np.array((start_position, stop_position)))
