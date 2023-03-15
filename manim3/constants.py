__all__ = [
    "DEGREES",
    "DL",
    "DOWN",
    "DR",
    "IN",
    "LEFT",
    "ORIGIN",
    "OUT",
    "PI",
    "RIGHT",
    "TAU",
    "UL",
    "UP",
    "UR",
    "X_AXIS",
    "Y_AXIS",
    "Z_AXIS"
]


import numpy as np

from .custom_typing import Vec3T


ORIGIN: Vec3T = np.array((0.0, 0.0, 0.0))
RIGHT: Vec3T = np.array((1.0, 0.0, 0.0))
LEFT: Vec3T = np.array((-1.0, 0.0, 0.0))
UP: Vec3T = np.array((0.0, 1.0, 0.0))
DOWN: Vec3T = np.array((0.0, -1.0, 0.0))
OUT: Vec3T = np.array((0.0, 0.0, 1.0))
IN: Vec3T = np.array((0.0, 0.0, -1.0))
X_AXIS: Vec3T = np.array((1.0, 0.0, 0.0))
Y_AXIS: Vec3T = np.array((0.0, 1.0, 0.0))
Z_AXIS: Vec3T = np.array((0.0, 0.0, 1.0))

UR: Vec3T = UP + RIGHT
UL: Vec3T = UP + LEFT
DL: Vec3T = DOWN + LEFT
DR: Vec3T = DOWN + RIGHT

PI: float = np.pi
TAU: float = PI * 2.0
DEGREES: float = PI / 180.0
