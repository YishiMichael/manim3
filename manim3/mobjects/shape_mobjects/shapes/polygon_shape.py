from ....constants.custom_typing import NP_x2f8
from ....utils.space_utils import SpaceUtils
from ...graph_mobjects.graphs.graph import Graph
from .shape import Shape


class PolygonShape(Shape):
    __slots__ = ()

    def __init__(
        self,
        positions: NP_x2f8
    ) -> None:
        super().__init__(Graph(
            positions=SpaceUtils.increase_dimension(positions),
            edges=Graph._get_consecutive_edges(len(positions), is_ring=True)
        ))
