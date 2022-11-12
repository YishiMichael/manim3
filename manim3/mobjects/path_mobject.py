import numpy as np
import skia

from ..mobjects.svg_mobject import SkiaPathMobject
from ..constants import ORIGIN
from ..typing import *


__all__ = ["PathMobject"]


class PathMobject(SkiaPathMobject):
    def __init__(
        self: Self,
        path: skia.Path,
        frame_buff: tuple[Real, Real] = (0.5, 0.5),
    ):
        super().__init__(path=path, frame_buff=frame_buff)
        self.scale(np.array((1.0, -1.0, 1.0)), about_point=ORIGIN)  # flip y
