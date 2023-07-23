from ....constants.custom_typing import NP_x2f8
from ....utils.space import SpaceUtils
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
            indices=Graph._get_consecutive_indices(len(positions), is_ring=True)
        ))
