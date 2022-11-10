import skia

from ..mobjects.skia_mobject import SkiaMobject
from ..typing import *


__all__ = ["PathMobject"]


class PathMobject(SkiaMobject):
    def __init__(self: Self, path: skia.Path, paint: skia.Paint, **kwargs):
        self.path: skia.Path = path
        self.paint: skia.Paint = paint
        skia_rect = path.computeTightBounds().makeOutset(0.0, 0.0) # TODO
        super().__init__(skia_rect.width(), skia_rect.height(), **kwargs)

    def draw(self: Self, canvas: skia.Canvas) -> None:
        canvas.drawPath(self.path, self.paint)
