from __future__ import annotations


from typing import Self

import numpy as np

from ...animatables.graph import Graph
from ...constants.custom_typing import NP_x3f8
from .graph_mobject import GraphMobject


class Polyline(GraphMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        positions: NP_x3f8
    ) -> None:
        arange = np.arange(len(positions))
        super().__init__(Graph(
            positions=positions,
            edges=np.vstack((arange[:-1], arange[1:])).T
        ))
