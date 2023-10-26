from __future__ import annotations


from typing import Self

import numpy as np

from ...constants.custom_typing import NP_f8
from .animatable_array import AnimatableArray


class AnimatableFloat(AnimatableArray[NP_f8]):
    __slots__ = ()

    def __init__(
        self: Self,
        value: float
    ) -> None:
        super().__init__(np.asarray(value, dtype=np.float64))

    #@Lazy.variable()
    #@staticmethod
    #def _array_() -> NP_f8:
    #    return np.zeros(())

    #@classmethod
    #def _convert_input(
    #    cls: type[Self],
    #    float_input: float
    #) -> Self:
    #    return AnimatableFloat(float_input)
