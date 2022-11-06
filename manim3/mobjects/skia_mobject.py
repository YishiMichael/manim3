import numpy as np
import skia

from ..constants import DEFAULT_PIXEL_HEIGHT, DEFAULT_PIXEL_WIDTH
from ..mobjects.planar_mobject import PlanarMobject
from ..typing import *


__all__ = ["SkiaMobject"]


class SkiaMobject(PlanarMobject):
    def init_color_map(self: Self) -> TextureArrayType:
        width = DEFAULT_PIXEL_WIDTH
        height = DEFAULT_PIXEL_HEIGHT
        array = np.zeros((height, width, 4), dtype=np.uint8)
        with skia.Surface(array) as canvas:
            canvas.concat(skia.Matrix.MakeRectToRect(
                skia.Rect(-1.0, -1.0, 1.0, 1.0),
                skia.Rect(0, 0, width, height),
                skia.Matrix.kFill_ScaleToFit
            ))
            self.draw(canvas)
        return array

    def draw(self: Self, canvas: skia.Canvas) -> None:
        pass
