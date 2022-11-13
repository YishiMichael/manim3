import numpy as np
import os

from .custom_typing import *


__all__ = [
    "MANIM3_PATH",
    "SHADERS_PATH",
    "PIXEL_HEIGHT",
    "PIXEL_WIDTH",
    "ASPECT_RATIO",
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "PIXEL_PER_UNIT",
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
SHADERS_PATH: str = os.path.join(MANIM3_PATH, "shaders")

PIXEL_HEIGHT: int = 1080
PIXEL_WIDTH: int = 1920
ASPECT_RATIO: float = PIXEL_WIDTH / PIXEL_HEIGHT

FRAME_HEIGHT: float = 8.0
FRAME_WIDTH: float = FRAME_HEIGHT * ASPECT_RATIO
PIXEL_PER_UNIT: float = PIXEL_HEIGHT / FRAME_HEIGHT
FRAME_Y_RADIUS: float = FRAME_HEIGHT / 2.0
FRAME_X_RADIUS: float = FRAME_WIDTH / 2.0
CAMERA_ALTITUDE: float = 2.0
CAMERA_NEAR: float = 0.1
CAMERA_FAR: float = 100.0

ORIGIN: Vector3Type = np.array((0.0, 0.0, 0.0))
RIGHT: Vector3Type = np.array((1.0, 0.0, 0.0))
LEFT: Vector3Type = np.array((-1.0, 0.0, 0.0))
UP: Vector3Type = np.array((0.0, 1.0, 0.0))
DOWN: Vector3Type = np.array((0.0, -1.0, 0.0))
OUT: Vector3Type = np.array((0.0, 0.0, 1.0))
IN: Vector3Type = np.array((0.0, 0.0, -1.0))
X_AXIS: Vector3Type = np.array((1.0, 0.0, 0.0))
Y_AXIS: Vector3Type = np.array((0.0, 1.0, 0.0))
Z_AXIS: Vector3Type = np.array((0.0, 0.0, 1.0))

UR: Vector3Type = UP + RIGHT
UL: Vector3Type = UP + LEFT
DL: Vector3Type = DOWN + LEFT
DR: Vector3Type = DOWN + RIGHT

PI: float = np.pi
TAU: float = PI * 2.0
DEGREES: float = PI / 180.0
