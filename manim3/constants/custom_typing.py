from __future__ import annotations


__all__ = (
    "NPE_f8",
    "NP_f8",
    "NP_2f8",
    "NP_3f8",
    "NP_4f8",
    "NP_44f8",
    "NP_xf8",
    "NP_x2f8",
    "NP_x3f8",
    "NP_x4f8",
    "NP_x33f8",
    "NP_x44f8",
    "NPE_i4",
    "NP_i4",
    "NP_xi4",
    "NP_x2i4",
    "NP_x3i4",
    "NP_xxi4",
    "BoundaryType",
    "ColorType",
    "AlignmentType",
    "SelectorType"
)


import re
from typing import Literal

import numpy as np
import numpy.typing as npt
from colour import Color


type NPE_f8 = np.float64
type NP_f8 = npt.NDArray[np.float64]
type NP_2f8 = npt.NDArray[np.float64]
type NP_3f8 = npt.NDArray[np.float64]
type NP_4f8 = npt.NDArray[np.float64]
type NP_44f8 = npt.NDArray[np.float64]
type NP_xf8 = npt.NDArray[np.float64]
type NP_x2f8 = npt.NDArray[np.float64]
type NP_x3f8 = npt.NDArray[np.float64]
type NP_x4f8 = npt.NDArray[np.float64]
type NP_x33f8 = npt.NDArray[np.float64]
type NP_x44f8 = npt.NDArray[np.float64]

type NPE_i4 = np.int32
type NP_i4 = npt.NDArray[np.int32]
type NP_xi4 = npt.NDArray[np.int32]
type NP_x2i4 = npt.NDArray[np.int32]
type NP_x3i4 = npt.NDArray[np.int32]
type NP_xxi4 = npt.NDArray[np.int32]

type BoundaryType = Literal[0, 1]
type ColorType = Color | str | NP_3f8
type AlignmentType = Literal["left", "center", "right"]
type SelectorType = str | re.Pattern[str] | slice
