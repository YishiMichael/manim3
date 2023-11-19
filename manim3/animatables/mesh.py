from __future__ import annotations


from typing import Self

import numpy as np

from ..constants.custom_typing import (
    NP_x2f8,
    NP_x3f8,
    NP_x3i4
)
from ..lazy.lazy import Lazy
from .animatable.animatable import Animatable


class Mesh(Animatable):
    __slots__ = ()

    def __init__(
        self: Self,
        positions: NP_x3f8 | None = None,
        normals: NP_x3f8 | None = None,
        uvs: NP_x2f8 | None = None,
        faces: NP_x3i4 | None = None
    ) -> None:
        super().__init__()
        if positions is not None:
            self._positions_ = positions
        if normals is not None:
            self._normals_ = normals
        if uvs is not None:
            self._uvs_ = uvs
        if faces is not None:
            self._faces_ = faces

    @Lazy.variable()
    @staticmethod
    def _positions_() -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.variable()
    @staticmethod
    def _normals_() -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.variable()
    @staticmethod
    def _uvs_() -> NP_x2f8:
        return np.zeros((0, 2))

    @Lazy.variable()
    @staticmethod
    def _faces_() -> NP_x3i4:
        return np.zeros((0, 3), dtype=np.int32)
