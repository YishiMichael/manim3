from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..custom_typing import *


__all__ = [
    "GeometryAttributes",
    "Geometry"
]


@dataclass
class GeometryAttributes:
    index: VertexIndicesType
    position: Vector3ArrayType
    uv: Vector2ArrayType


class Geometry(ABC):
    def __init__(self: Self):
        attributes = self.init_geometry_attributes()
        self.index: VertexIndicesType = attributes.index
        self.position: Vector3ArrayType = attributes.position
        self.uv: Vector2ArrayType = attributes.uv

    @abstractmethod
    def init_geometry_attributes(self: Self) -> GeometryAttributes:
        raise NotImplementedError
