#import numpy as np
#
#from ..mobjects.plane import Plane
#from ..typing import *
#
#
#__all__ = ["PlanarMobject"]
#
#
#class PlanarMobject(Plane):
#    def __init__(self: Self, width: Real, height: Real):
#        #self.rectangle: BoundingBox2D = rectangle
#        self.width: float = float(width)
#        self.height: float = float(height)
#        super().__init__(width, height, color_map=self.init_color_map())
#        self.scale(np.array((1.0, -1.0, 1.0)))  # flip y
#        self.enable_depth_test = False
#        # The object is of square shape at [-1, 1] x [-1, 1].
#        # Scale and shift as specified by `rectangle`.
#        #self.scale(np.array((*rectangle.radius, 1.0)))
#        #self.shift(np.array((*rectangle.origin, 0.0)))
#
#    def init_color_map(self: Self) -> TextureArrayType:
#        raise NotImplementedError
