from __future__ import annotations


__all__ = (
    "NP_f8",
    "NP_2f8",
    "NP_3f8",
    "NP_44f8",
    "NP_xf8",
    "NP_x2f8",
    "NP_x3f8",
    "NP_xi4",
    "NP_x2i4",
    "NP_x3i4",
    "NP_xxi4",
    "ShapeType",
    "ColorType",
    "AlignmentType",
    "SelectorType",
    "RateType",
    "ConditionType"
)


import re
from typing import (
    Callable,
    Literal
)

import numpy as np
from colour import Color


type _XD = int
type _2D = Literal[2]
type _3D = Literal[3]
type _4D = Literal[4]

type NP_f8 = np.ndarray[tuple[()], np.dtype[np.float64]]
type NP_2f8 = np.ndarray[tuple[_2D], np.dtype[np.float64]]
type NP_3f8 = np.ndarray[tuple[_3D], np.dtype[np.float64]]
type NP_44f8 = np.ndarray[tuple[_4D, _4D], np.dtype[np.float64]]
type NP_xf8 = np.ndarray[tuple[_XD], np.dtype[np.float64]]
type NP_x2f8 = np.ndarray[tuple[_XD, _2D], np.dtype[np.float64]]
type NP_x3f8 = np.ndarray[tuple[_XD, _3D], np.dtype[np.float64]]

type NP_xi4 = np.ndarray[tuple[_XD], np.dtype[np.int32]]
type NP_x2i4 = np.ndarray[tuple[_XD, _2D], np.dtype[np.int32]]
type NP_x3i4 = np.ndarray[tuple[_XD, _3D], np.dtype[np.int32]]
type NP_xxi4 = np.ndarray[tuple[_XD, _XD], np.dtype[np.int32]]

type ShapeType = tuple[int, ...]
type ColorType = Color | str | NP_3f8
type AlignmentType = Literal["left", "center", "right"]
type SelectorType = str | re.Pattern[str] | slice

type RateType = Callable[[float], float]
type ConditionType = Callable[[], bool]
