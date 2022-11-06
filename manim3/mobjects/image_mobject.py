import numpy as np
from PIL import Image

from ..mobjects.planar_mobject import PlanarMobject
from ..typing import *


__all__ = ["ImageMobject"]


class ImageMobject(PlanarMobject):
    def __init__(self: Self, image_path: str, **kwargs):
        self.image_path: str = image_path
        super().__init__(**kwargs)

    def init_color_map(self: Self) -> TextureArrayType:
        img = Image.open(self.image_path).convert("RGBA")
        return np.array(img)[::-1, :]  # Flip y axis
