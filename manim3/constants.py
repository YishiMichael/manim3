import numpy as np
import os


__all__ = [
    "MANIM3_PATH",
    "DEFAULT_PIXEL_HEIGHT",
    "DEFAULT_PIXEL_WIDTH",
    "ASPECT_RATIO",
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "FRAME_Y_RADIUS",
    "FRAME_X_RADIUS",
    "CAMERA_ALTITUDE",
    "CAMERA_NEAR",
    "CAMERA_FAR",
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
    "DEGREES",
]


MANIM3_PATH: str = os.path.dirname(__file__)

DEFAULT_PIXEL_HEIGHT: int = 1080
DEFAULT_PIXEL_WIDTH: int = 1920
ASPECT_RATIO: float = DEFAULT_PIXEL_WIDTH / DEFAULT_PIXEL_HEIGHT

FRAME_HEIGHT: float = 8.0
FRAME_WIDTH: float = FRAME_HEIGHT * ASPECT_RATIO
FRAME_Y_RADIUS: float = FRAME_HEIGHT / 2
FRAME_X_RADIUS: float = FRAME_WIDTH / 2
CAMERA_ALTITUDE: float = 2.0
CAMERA_NEAR: float = 0.1
CAMERA_FAR: float = 100.0

ORIGIN = np.array((0.0, 0.0, 0.0))
RIGHT = np.array((1.0, 0.0, 0.0))
LEFT = np.array((-1.0, 0.0, 0.0))
UP = np.array((0.0, 1.0, 0.0))
DOWN = np.array((0.0, -1.0, 0.0))
OUT = np.array((0.0, 0.0, 1.0))
IN = np.array((0.0, 0.0, -1.0))
X_AXIS = np.array((1.0, 0.0, 0.0))
Y_AXIS = np.array((0.0, 1.0, 0.0))
Z_AXIS = np.array((0.0, 0.0, 1.0))

UR = UP + RIGHT
UL = UP + LEFT
DL = DOWN + LEFT
DR = DOWN + RIGHT

PI = np.pi
TAU = PI * 2.0
DEGREES = PI / 180.0
