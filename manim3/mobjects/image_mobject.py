import skia

from ..mobjects.skia_mobject import SkiaMobject
from ..utils.paint import Paint
from ..constants import PIXEL_PER_UNIT
from ..custom_typing import *


__all__ = ["ImageMobject"]


class ImageMobject(SkiaMobject):
    def __init__(
        self: Self,
        image_path: str,
        paint: Paint | None = None,
        *,
        width: Real | None = None,
        height: Real | None = None,
        frame_scale: Real | None = None
    ):
        super().__init__(
            #frame=frame,
            #resolution=(px_width, px_height)
        )
        image = skia.Image.open(image_path).convert(
            colorType=skia.kRGBA_8888_ColorType,
            alphaType=skia.kUnpremul_AlphaType
        )
        self.image: skia.Image = image
        self.paint: Paint | None = paint

        self._frame: skia.Rect = self.calculate_frame_size(
            image.width() / PIXEL_PER_UNIT,
            image.height() / PIXEL_PER_UNIT,
            width,
            height,
            frame_scale
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

    @property
    def frame(self: Self) -> skia.Rect:
        return self._frame

    @property
    def resolution(self: Self) -> tuple[int, int]:
        image = self.image
        return (image.width(), image.height())

    def draw(self: Self, canvas: skia.Canvas) -> None:
        canvas.drawImage(self.image, 0.0, 0.0, self.paint)
