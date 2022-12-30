__all__ = ["ImageMobject"]


import moderngl
import skia

from ..mobjects.skia_mobject import SkiaMobject
from ..utils.lazy import lazy_property, lazy_property_initializer_writable
from ..utils.paint import Paint
from ..constants import PIXEL_PER_UNIT
from ..custom_typing import *


class ImageMobject(SkiaMobject):
    def __init__(
        self,
        image_path: str,
        paint: Paint | None = None,
        *,
        width: Real | None = None,
        height: Real | None = None,
        frame_scale: Real | None = None
    ):
        super().__init__()
        image = skia.Image.open(image_path).convert(
            colorType=skia.kRGBA_8888_ColorType,
            alphaType=skia.kUnpremul_AlphaType
        )
        self._image_ = image
        self._paint_ = paint
        self._frame_ = self._calculate_frame(
            image.width() / PIXEL_PER_UNIT,
            image.height() / PIXEL_PER_UNIT,
            width,
            height,
            frame_scale
        )

    @lazy_property_initializer_writable
    @classmethod
    def _image_(cls) -> skia.Image:
        return NotImplemented

    @lazy_property_initializer_writable
    @classmethod
    def _paint_(cls) -> Paint | None:
        return None

    @lazy_property_initializer_writable
    @classmethod
    def _frame_(cls) -> skia.Rect:
        return NotImplemented

    @lazy_property
    @classmethod
    def _uniform_color_map_texture_(
        cls,
        image: skia.Image,
        paint: Paint | None
    ) -> moderngl.Texture:
        surface = cls._make_surface(image.width(), image.height())
        with surface as canvas:
            canvas.drawImage(
                image=image, left=0.0, top=0.0, paint=paint
            )
        return cls._make_texture(surface.makeImageSnapshot())
