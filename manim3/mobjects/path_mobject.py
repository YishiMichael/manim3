#from functools import reduce
import numpy as np
#import pyrr
import skia

from ..mobjects.svg_mobject import SkiaPathMobject
from ..constants import ORIGIN
from ..typing import *


__all__ = ["PathMobject"]


class PathMobject(SkiaPathMobject):
    def __init__(
        self: Self,
        path: skia.Path,
        transform_matrix: skia.Matrix | None = None,
        frame_buff: tuple[Real, Real] = (0.5, 0.5),
    ):
        if transform_matrix is None:
            transform_matrix = skia.Matrix.I()
        super().__init__(path=path, transform_matrix=transform_matrix, frame_buff=frame_buff)
        self.scale(np.array((1.0, -1.0, 1.0)), about_point=ORIGIN)  # flip y
