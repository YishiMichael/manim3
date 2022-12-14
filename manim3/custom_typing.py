__all__ = [
    "Real",
    "Vec2T",
    "Vec3T",
    "Vec4T",
    "Mat3T",
    "Mat4T",
    "FloatsT",
    "Vec2sT",
    "Vec3sT",
    "Vec4sT",
    "Mat3sT",
    "Mat4sT",
    "VertexIndicesType",
    "ColorType",
    "Span",
    "Selector"
]


from colour import Color
import re
from typing import (
    Iterable,
    Literal,
    Union
)

import numpy as np


Real = float | int

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

VertexIndicesType = np.ndarray[tuple[_ND], np.dtype[np.uint]]

ColorType = Color | str
Span = tuple[int, int]
Selector = Union[
    str,
    re.Pattern,
    tuple[Union[int, None], Union[int, None]],
    Iterable[Union[
        str,
        re.Pattern,
        tuple[Union[int, None], Union[int, None]]
    ]]
]
