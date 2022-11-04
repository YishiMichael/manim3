from cameras.camera import Camera
from materials.material import Material
from utils.typing import *


class BasicMaterial(Material):
    def __init__(self: Self, color: ColorArrayType):
        super().__init__()
        self.color: ColorArrayType = color

    def get_define_macros(self: Self) -> list[str]:
        return []

    def get_uniforms(self: Self, camera: Camera) -> dict[str, UniformType]:
        return {
            "uniform_transform_matrix": camera.get_transform_matrix(),
            "uniform_color": self.color,
        }

    def get_texture_arrays(self: Self) -> list[TextureArrayType]:
        return []
