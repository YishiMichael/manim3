from abc import ABC, abstractmethod

from utils.typing import *


__all__ = ["Geometry"]


class Geometry(ABC):
    @abstractmethod
    def get_vertex_indices(self: Self) -> VertexIndicesType:
        raise NotImplementedError

    @abstractmethod
    def get_attributes_v(self: Self) -> AttributesItemType:
        raise NotImplementedError
