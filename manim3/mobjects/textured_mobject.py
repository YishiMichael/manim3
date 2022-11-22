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

    def load_color_map(self: Self) -> skia.Image | None:
        return self.image
        #image = self.image
        #if image is not None:
        #    info = image.imageInfo()
        #    pixmap = skia.Pixmap(info, None, info.width() * info.bytesPerPixel())
        #    image.readPixels(pixmap)  # TODO: test (try using ctx)
        #    return pixmap
        #return None
