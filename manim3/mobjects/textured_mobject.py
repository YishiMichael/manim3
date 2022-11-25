import moderngl
import skia

from ..mobjects.mesh_mobject import MeshMobject
from ..utils.lazy import lazy_property_initializer
from ..custom_typing import *


__all__ = ["TexturedMobject"]


class TexturedMobject(MeshMobject):
    def __init__(
        self,
        image_path: str | None = None
    ):
        super().__init__()
        if image_path is not None:
            color_map = self._make_texture(
                skia.Image.open(image_path).convert(
                    colorType=skia.kRGBA_8888_ColorType,
                    alphaType=skia.kUnpremul_AlphaType
                )
            )
        else:
            color_map = None
        self._color_map_ = color_map

    @lazy_property_initializer
    def _color_map_() -> moderngl.Texture | None:
        raise NotImplementedError
        #image = self.image
        #if image is not None:
        #    info = image.imageInfo()
        #    pixmap = skia.Pixmap(info, None, info.width() * info.bytesPerPixel())
        #    image.readPixels(pixmap)  # TODO: test (try using ctx)
        #    return pixmap
        #return None
