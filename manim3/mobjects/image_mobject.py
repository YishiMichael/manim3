import moderngl
import skia

from ..mobjects.skia_mobject import SkiaMobject
from ..utils.lazy import lazy_property, lazy_property_initializer
from ..utils.paint import Paint
from ..constants import PIXEL_PER_UNIT
from ..custom_typing import *


__all__ = ["ImageMobject"]


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
        super().__init__(
            #frame=frame,
            #resolution=(px_width, px_height)
        )
        #px_width = image.width()
        #px_height = image.height()
        #frame = self.calculate_frame(
        #    px_width / PIXEL_PER_UNIT,
        #    px_height / PIXEL_PER_UNIT,
        #    width,
        #    height,
        #    frame_scale
        #)
        #self.frame_size: tuple[float, float] = self.calculate_frame_size(
        #    image.width() / PIXEL_PER_UNIT,
        #    image.height() / PIXEL_PER_UNIT,
        #    width,
        #    height,
        #    frame_scale
        #)

    @lazy_property_initializer
    def _image_() -> skia.Image:
        return NotImplemented

    @lazy_property_initializer
    def _paint_() -> Paint | None:
        return None

    @lazy_property_initializer
    def _frame_() -> skia.Rect:
        return NotImplemented

    @lazy_property
    def _color_map_(
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

    #@lazy_property
    #def _resolution_(image: skia.Image) -> tuple[int, int]:
    #    return (image.width(), image.height())

    #@lazy_property
    #def _draw_(
    #    image: skia.Image,
    #    paint: Paint | None
    #) -> Callable[[skia.Canvas], None]:
    #    def draw(canvas: skia.Canvas) -> None:
    #        canvas.drawImage(image, 0.0, 0.0, paint)
    #    return draw
