__all__ = (
    "ORIGIN",
    "RIGHT",
    "LEFT",
    "UP",
    "DOWN",
    "OUT",
    "IN",
    "X_AXIS",
    "Y_AXIS",
    "Z_AXIS",
    "UR",
    "UL",
    "DL",
    "DR",
    "PI",
    "TAU",
    "DEGREES"
)


import numpy as np

from .custom_typing import NP_3f8


ORIGIN: NP_3f8 = np.array((0.0, 0.0, 0.0))
RIGHT: NP_3f8 = np.array((1.0, 0.0, 0.0))
LEFT: NP_3f8 = np.array((-1.0, 0.0, 0.0))
UP: NP_3f8 = np.array((0.0, 1.0, 0.0))
DOWN: NP_3f8 = np.array((0.0, -1.0, 0.0))
OUT: NP_3f8 = np.array((0.0, 0.0, 1.0))
IN: NP_3f8 = np.array((0.0, 0.0, -1.0))
X_AXIS: NP_3f8 = np.array((1.0, 0.0, 0.0))
Y_AXIS: NP_3f8 = np.array((0.0, 1.0, 0.0))
Z_AXIS: NP_3f8 = np.array((0.0, 0.0, 1.0))

UR: NP_3f8 = UP + RIGHT
UL: NP_3f8 = UP + LEFT
DL: NP_3f8 = DOWN + LEFT
DR: NP_3f8 = DOWN + RIGHT

PI: float = np.pi
TAU: float = PI * 2.0
DEGREES: float = PI / 180.0
