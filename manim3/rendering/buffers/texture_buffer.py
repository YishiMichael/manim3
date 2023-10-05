from __future__ import annotations


import re
from typing import Self

import moderngl
import numpy as np

from ...lazy.lazy import Lazy
from .buffer import Buffer


class TextureBuffer(Buffer):
    __slots__ = ()

    def __init__(
        self: Self,
        *,
        field: str,
        array_lens: dict[str, int] | None = None,
        # Note, redundant textures are currently not supported.
        texture_array: np.ndarray
    ) -> None:
        replaced_field = re.sub(r"^sampler2D\b", "uint", field)
        assert field != replaced_field
        super().__init__(
            field=replaced_field,
            child_structs=None,
            array_lens=array_lens
        )
        self._texture_array_ = texture_array

    @Lazy.variable()
    @staticmethod
    def _texture_array_() -> np.ndarray:
        return np.zeros((), dtype=moderngl.Texture)
