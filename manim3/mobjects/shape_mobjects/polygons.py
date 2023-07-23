#import numpy as np

#from ...constants.constants import (
#    OUT,
#    PI,
#    TAU
#)
#from ...constants.custom_typing import NP_x2f8
#from ..mobject.shape.shape import Shape
#from ..mobject.shape.stroke import Stroke
#from ..shape_mobject import ShapeMobject


#class Polygon(ShapeMobject):
#    __slots__ = ()

#    def __init__(
#        self,
#        points: NP_x2f8
#    ) -> None:
#        super().__init__(Shape(Stroke(points=np.append(points, points[:1], axis=0))))


#class RegularPolygon(Polygon):
#    __slots__ = ()

#    def __init__(
#        self,
#        n: int
#    ) -> None:
#        # By default, one of positions is at (1, 0).
#        complex_coords = np.exp(1.0j * np.linspace(0.0, TAU, n, endpoint=False))
#        super().__init__(np.vstack((complex_coords.real, complex_coords.imag)).T)


#class Triangle(RegularPolygon):  # TODO: remove
#    __slots__ = ()

#    def __init__(self) -> None:
#        super().__init__(3)
#        self.rotate(PI / 2.0 * OUT)


#class Square(RegularPolygon):  # TODO
#    __slots__ = ()

#    def __init__(self) -> None:
#        super().__init__(4)
#        self.rotate(PI / 4.0 * OUT)


#class Circle(RegularPolygon):
#    __slots__ = ()

#    def __init__(self) -> None:
#        super().__init__(64)


#class Arc(Polygon):
#    __slots__ = ()

#    def __init__(
#        self,
#        start_angle: float,
#        sweep_angle: float
#    ) -> None:
#        n_segments = int(np.ceil(sweep_angle / TAU * 64.0))
#        complex_coords = np.exp(1.0j * (start_angle + np.linspace(0.0, sweep_angle, n_segments + 1)))
#        super().__init__(np.vstack((complex_coords.real, complex_coords.imag)).T)
