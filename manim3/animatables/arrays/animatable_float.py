from __future__ import annotations


from typing import Self

import numpy as np

from ...constants.custom_typing import NP_f8
from ...lazy.lazy import Lazy
from .animatable_array import AnimatableArray


class AnimatableFloat(AnimatableArray[NP_f8]):
    __slots__ = ()

    def __init__(
        self: Self,
        value: float | NP_f8 | None = None
    ) -> None:
        super().__init__(value if isinstance(value, np.ndarray | None) else np.asarray(value))

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _array_() -> NP_f8:
        return np.zeros(())

    @classmethod
    def _convert_input(
        cls: type[Self],
        float_input: float | np.ndarray
    ) -> Self:
        array = float_input if isinstance(float_input, np.ndarray) else np.asarray(float_input)
        return super()._convert_input(array.reshape(()).astype(np.float64))
