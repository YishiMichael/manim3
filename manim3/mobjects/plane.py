#import numpy as np
#
#from ..mobjects.parametric_surface import ParametricSurface
#from ..typing import *
#
#
#__all__ = ["Plane"]
#
#
#class Plane(ParametricSurface):
#    def __init__(
#        self: Self,
#        width: Real = 2.0,
#        height: Real = 2.0,
#        width_segments: int = 1,
#        height_segments: int = 1,
#        **kwargs
#    ):
#        super().__init__(
#            lambda x, y: np.array((x, y, 0.0)),
#            (-width / 2.0, width / 2.0),
#            (-height / 2.0, height / 2.0),
#            resolution=(width_segments, height_segments),
#            **kwargs
#        )
