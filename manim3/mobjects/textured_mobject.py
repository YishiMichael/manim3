import skia

from ..mobjects.mesh_mobject import MeshMobject
from ..custom_typing import *


__all__ = ["TexturedMobject"]


class TexturedMobject(MeshMobject):
    def __init__(
        self: Self,
        image_path: str | None = None
    ):
        super().__init__()
        if image_path is not None:
            image = skia.Image.open(image_path).convert(
                colorType=skia.kRGBA_8888_ColorType,
                alphaType=skia.kUnpremul_AlphaType
            )
        else:
            image = None
        self.image: skia.Image | None = image

    def load_color_map(self: Self) -> TextureArrayType | None:
        if self.image is not None:
            return self.image.toarray()
        return None
