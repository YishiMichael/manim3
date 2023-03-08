__all__ = [
    "ColorType",
    "FloatsT",
    "Mat3T",
    "Mat4T",
    "Mat3sT",
    "Mat4sT",
    "Selector",
    "Vec2T",
    "Vec3T",
    "Vec4T",
    "Vec2sT",
    "Vec3sT",
    "Vec4sT",
    "VertexIndexType"
]


from colour import Color
import re
from typing import (
    Iterable,
    Literal
)

import numpy as np


_ND = int
_2D = Literal[2]
_3D = Literal[3]
_4D = Literal[4]

Vec2T = np.ndarray[tuple[_2D], np.dtype[np.float_]]
Vec3T = np.ndarray[tuple[_3D], np.dtype[np.float_]]
Vec4T = np.ndarray[tuple[_4D], np.dtype[np.float_]]
Mat3T = np.ndarray[tuple[_3D, _3D], np.dtype[np.float_]]
Mat4T = np.ndarray[tuple[_4D, _4D], np.dtype[np.float_]]

FloatsT = np.ndarray[tuple[_ND], np.dtype[np.float_]]
Vec2sT = np.ndarray[tuple[_ND, _2D], np.dtype[np.float_]]
Vec3sT = np.ndarray[tuple[_ND, _3D], np.dtype[np.float_]]
Vec4sT = np.ndarray[tuple[_ND, _4D], np.dtype[np.float_]]
Mat3sT = np.ndarray[tuple[_ND, _3D, _3D], np.dtype[np.float_]]
Mat4sT = np.ndarray[tuple[_ND, _4D, _4D], np.dtype[np.float_]]

VertexIndexType = np.ndarray[tuple[_ND], np.dtype[np.uint]]

ColorType = Color | str | Vec3T | Vec4T
Selector = str | re.Pattern[str] | slice | Iterable[str | re.Pattern[str] | slice]
