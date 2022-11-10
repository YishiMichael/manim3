import numpy as np
import skia

from ..mobjects.skia_mobject import BoundingBox2D, SkiaMobject
from ..typing import *


__all__ = ["PathMobject"]


class PathMobject(SkiaMobject):
    def __init__(
        self: Self,
        path: skia.Path,
        paint: skia.Paint | None = None,
        frame_buff: float = 0.5
    ):
        self.path: skia.Path = path
        self.paint: skia.Paint | None = paint
        rect = path.computeTightBounds().makeOutset(frame_buff, frame_buff)
        frame = BoundingBox2D(
            origin=np.array((rect.centerX(), rect.centerY())),
            radius=np.array((rect.width(), rect.height())) / 2.0
        )
        super().__init__(frame=frame)

    def draw(self: Self, canvas: skia.Canvas) -> None:
        canvas.drawPath(self.path, self.paint)
