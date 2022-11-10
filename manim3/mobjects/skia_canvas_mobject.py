#import numpy as np
#import pyrr
#
#from ..mobjects.skia_base_mobject import BoundingBox2D, SkiaBaseMobject
#from ..constants import PIXEL_PER_UNIT
#from ..typing import *
#
#
#__all__ = ["SkiaCanvasMobject"]
#
#
#class SkiaCanvasMobject(SkiaBaseMobject):
#    def __init__(
#        self: Self,
#        #draw_range: BoundingBox2D,
#        frame: BoundingBox2D,
#        resolution: tuple[int, int] | None = None,
#        #canvas_matrix: Matrix33Type,
#        #geometry: Geometry | None = None,
#        #rectangle: BoundingBox2D,
#        #resolution: tuple[int, int] = (PIXEL_WIDTH, PIXEL_HEIGHT)
#    ):
#        if resolution is None:
#            arr = abs(2.0 * PIXEL_PER_UNIT * frame.radius)
#            resolution = (int(arr[0]), int(arr[1]))
#        super().__init__(resolution, frame)
#
#    def get_canvas_matrix(self: Self) -> pyrr.Matrix33 | None:
