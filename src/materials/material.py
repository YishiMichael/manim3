from abc import ABC, abstractmethod

from cameras.camera import Camera
from utils.typing import *


__all__ = ["Material"]


class Material(ABC):
    @abstractmethod
    def get_define_macros(self: Self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_uniforms(self: Self, camera: Camera) -> dict[str, UniformType]:
        raise NotImplementedError

    @abstractmethod
    def get_texture_arrays(self: Self) -> list[TextureArrayType]:
        raise NotImplementedError
