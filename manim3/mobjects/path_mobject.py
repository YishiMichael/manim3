import numpy as np
import skia

from ..mobjects.skia_mobject import SkiaMobject
from ..constants import ORIGIN
from ..constants import PIXEL_PER_UNIT
from ..typing import *


__all__ = ["PathMobject"]


class PathMobject(SkiaMobject):
    def __init__(
        self: Self,
        path: skia.Path,
        frame_buff: tuple[Real, Real] = (0.5, 0.5),
        flip_y: bool = True
    ):
        frame = path.computeTightBounds().makeOutset(*frame_buff)
        super().__init__(
            frame=frame,
            resolution=(
                int(frame.width() * PIXEL_PER_UNIT),
                int(frame.height() * PIXEL_PER_UNIT)
            )
        )
        if flip_y:
            self.scale(np.array((1.0, -1.0, 1.0)), about_point=ORIGIN)

        self.path: skia.Path = path
        self.fill_paint: skia.Paint | None = None
        self.stroke_paint: skia.Paint | None = None

    def draw(self: Self, canvas: skia.Canvas) -> None:
        canvas.concat(skia.Matrix.MakeRectToRect(
            self.frame, skia.Rect.MakeWH(*self.resolution), skia.Matrix.kFill_ScaleToFit
        ))
        if self.fill_paint is not None:
            canvas.drawPath(self.path, self.fill_paint)
        if self.stroke_paint is not None:
            canvas.drawPath(self.path, self.stroke_paint)
