import numpy as np
import skia

from ..mobjects.skia_mobject import SkiaMobject
from ..typing import *


__all__ = ["ImageMobject"]


class ImageMobject(SkiaMobject):
    def __init__(
        self: Self,
        image_path: str,
        #width: Real | None = None,
        #height: Real | None = 4.0,
        paint: skia.Paint | None = None,
        *,
        width: Real | None = None,
        height: Real | None = None
    ):
        image = skia.Image.open(image_path).convert(
            colorType=skia.kRGBA_8888_ColorType,
            alphaType=skia.kUnpremul_AlphaType
        )

        px_width = image.width()
        px_height = image.height()
        frame = self.calculate_frame_by_aspect_ratio(
            width, height, px_width / px_height
        )
        #aspect_ratio = px_width / px_height
        #if width is None and height is None:
        #    frame_size = None
        #else:
        #    if height is not None:
        #        width = height * aspect_ratio
        #    elif width is not None:
        #        height = width / aspect_ratio
        #    frame_size = np.array((width, height))
        super().__init__(
            frame=frame,
            resolution=(px_width, px_height)
        )
        self.image: skia.Image = image
        self.paint: skia.Paint | None = paint

    def draw(self: Self, canvas: skia.Canvas) -> None:
        canvas.drawImage(self.image, 0.0, 0.0, self.paint)
