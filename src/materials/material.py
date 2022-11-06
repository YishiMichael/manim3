from abc import ABC, abstractmethod

from cameras.camera import Camera
from utils.typing import *


__all__ = ["Material"]


class Material(ABC):
    @abstractmethod
    def get_define_macros(self: Self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_attributes_r(self: Self, camera: Camera) -> AttributesItemType:
        raise NotImplementedError

    @abstractmethod
    def get_texture_dict(self: Self) -> dict[int, TextureArrayType | None]:
        raise NotImplementedError
