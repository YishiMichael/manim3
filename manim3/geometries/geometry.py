from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..custom_typing import *


__all__ = [
    "GeometryAttributes",
    "Geometry"
]


@dataclass
class GeometryAttributes:
    indices: VertexIndicesType
    positions: Vector3ArrayType
    uvs: Vector2ArrayType


class Geometry(ABC):
    def __init__(self):
        attributes = self.init_geometry_attributes()
        self.indices: VertexIndicesType = attributes.indices
        self.positions: Vector3ArrayType = attributes.positions
        self.uvs: Vector2ArrayType = attributes.uvs

    @abstractmethod
    def init_geometry_attributes(self) -> GeometryAttributes:
        raise NotImplementedError
