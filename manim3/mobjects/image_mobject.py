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
        image = skia.Image.open(image_path).convert(
            colorType=skia.kRGBA_8888_ColorType,
            alphaType=skia.kUnpremul_AlphaType
        )

        px_width = image.width()
        px_height = image.height()
        frame = self.calculate_frame(
            px_width / PIXEL_PER_UNIT,
            px_height / PIXEL_PER_UNIT,
            width,
            height,
            frame_scale
        )
        super().__init__(
            frame=frame,
            resolution=(px_width, px_height)
        )
        self.image: skia.Image = image
        self.paint: Paint | None = paint

    def draw(self: Self, canvas: skia.Canvas) -> None:
        canvas.drawImage(self.image, 0.0, 0.0, self.paint)
