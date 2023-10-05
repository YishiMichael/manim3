from __future__ import annotations


import numpy as np

from ...constants.custom_typing import (
    NP_2f8,
    NP_44f8,
    NP_f8
)
from ...lazy.lazy import Lazy
from .camera import Camera


class OrthographicCamera(Camera):
    __slots__ = ()

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _projection_matrix_(
        frame_radii: NP_2f8,
        near: NP_f8,
        far: NP_f8
    ) -> NP_44f8:
        sx, sy = 1.0 / frame_radii
        sz = -2.0 / (far - near)
        tz = -(far + near) / (far - near)
        return np.array((
            ( sx, 0.0, 0.0, 0.0),
            (0.0,  sy, 0.0, 0.0),
            (0.0, 0.0,  sz,  tz),
            (0.0, 0.0, 0.0, 1.0)
        ))
