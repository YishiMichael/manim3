import moderngl
import skia

from ..mobjects.mesh_mobject import MeshMobject
from ..utils.lazy import lazy_property_initializer_writable
from ..custom_typing import *


__all__ = ["TexturedMobject"]


class TexturedMobject(MeshMobject):
    def __init__(
        self,
        image_path: str | None = None
    ):
        super().__init__()
        if image_path is not None:
            self._color_map_ = self._make_texture(
                skia.Image.open(image_path).convert(
                    colorType=skia.kRGBA_8888_ColorType,
                    alphaType=skia.kUnpremul_AlphaType
                )
            )

    @lazy_property_initializer_writable
    @staticmethod
    def _color_map_() -> moderngl.Texture | None:
        return None
        #image = self.image
        #if image is not None:
        #    info = image.imageInfo()
        #    pixmap = skia.Pixmap(info, None, info.width() * info.bytesPerPixel())
        #    image.readPixels(pixmap)  # TODO: test (try using ctx)
        #    return pixmap
        #return None
