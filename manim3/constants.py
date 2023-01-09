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
    "GLSL_DTYPE"
]


import numpy as np
import os

from .custom_typing import Vec3T


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

GLSL_DTYPE: dict[str, np.dtype] = {
    "int":     np.dtype(("i4", ())),
    "ivec2":   np.dtype(("i4", (2,))),
    "ivec3":   np.dtype(("i4", (3,))),
    "ivec4":   np.dtype(("i4", (4,))),
    "uint":    np.dtype(("u4", ())),
    "uvec2":   np.dtype(("u4", (2,))),
    "uvec3":   np.dtype(("u4", (3,))),
    "uvec4":   np.dtype(("u4", (4,))),
    "float":   np.dtype(("f4", ())),
    "vec2":    np.dtype(("f4", (2,))),
    "vec3":    np.dtype(("f4", (3,))),
    "vec4":    np.dtype(("f4", (4,))),
    "double":  np.dtype(("f8", ())),
    "dvec2":   np.dtype(("f8", (2,))),
    "dvec3":   np.dtype(("f8", (3,))),
    "dvec4":   np.dtype(("f8", (4,))),
    "mat2":    np.dtype(("f4", (2, 2))),
    "mat2x3":  np.dtype(("f4", (2, 3))),  # TODO: check order
    "mat2x4":  np.dtype(("f4", (2, 4))),
    "mat3x2":  np.dtype(("f4", (3, 2))),
    "mat3":    np.dtype(("f4", (3, 3))),
    "mat3x4":  np.dtype(("f4", (3, 4))),
    "mat4x2":  np.dtype(("f4", (4, 2))),
    "mat4x3":  np.dtype(("f4", (4, 3))),
    "mat4":    np.dtype(("f4", (4, 4))),
    "dmat2":   np.dtype(("f8", (2, 2))),
    "dmat2x3": np.dtype(("f8", (2, 3))),
    "dmat2x4": np.dtype(("f8", (2, 4))),
    "dmat3x2": np.dtype(("f8", (3, 2))),
    "dmat3":   np.dtype(("f8", (3, 3))),
    "dmat3x4": np.dtype(("f8", (3, 4))),
    "dmat4x2": np.dtype(("f8", (4, 2))),
    "dmat4x3": np.dtype(("f8", (4, 3))),
    "dmat4":   np.dtype(("f8", (4, 4))),
}
