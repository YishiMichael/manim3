import numpy as np

from cameras.camera import Camera
from materials.material import Material
from utils.typing import *


class BasicMaterial(Material):
    def __init__(
        self: Self,
        color: ColorArrayType | None = None,
        map: TextureArrayType | None = None
    ):
        super().__init__()
        if color is None:
            color = np.ones(4)
        self.color: ColorArrayType = color
        self.map: TextureArrayType | None = map

    def get_define_macros(self: Self) -> list[str]:
        result = []
        if self.map is not None:
            result.append("USE_UV")
            result.append("USE_MAP")
        return result

    def get_attributes_r(self: Self, camera: Camera) -> AttributesItemType:
        return np.array([(
            camera.get_projection_matrix(),
            camera.get_view_matrix(),
            self.color,
            0
        )], dtype=np.dtype([
            ("in_projection_matrix", (np.float32, (4, 4))),
            ("in_view_matrix", (np.float32, (4, 4))),
            ("in_color", (np.float32, (4,))),
            ("in_map", (np.int8, (1,)))
        ]))

    def get_texture_dict(self: Self) -> dict[int, TextureArrayType | None]:
        return {
            0: self.map
        }
