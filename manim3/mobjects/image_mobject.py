import numpy as np
import skia

from ..mobjects.skia_base_mobject import BoundingBox2D, SkiaBaseMobject
from ..typing import *


__all__ = ["ImageMobject"]


class ImageMobject(SkiaBaseMobject):
    def __init__(
        self: Self,
        image_path: str,
        height: float | None = 4.0,
        paint: skia.Paint | None = None
    ):
        image = skia.Image.open(image_path).convert(
            colorType=skia.kRGBA_8888_ColorType,
            alphaType=skia.kUnpremul_AlphaType
        )
        self.image: skia.Image = image
        self.paint: skia.Paint | None = paint
        px_width = image.width()
        px_height = image.height()
        if height is not None:
            frame = BoundingBox2D(
                origin=np.array((0.0, 0.0)),
                radius=np.array((height / px_height * px_width, height)) / 2.0
            )
        else:
            frame = None
        super().__init__((px_width, px_height), frame)

    def draw(self: Self, canvas: skia.Canvas) -> None:
        canvas.drawImage(self.image, 0.0, 0.0, self.paint)
        #return np.array(self.image)
