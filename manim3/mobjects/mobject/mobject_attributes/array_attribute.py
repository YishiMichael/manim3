from typing import (
    Generic,
    TypeVar
)

import numpy as np

from ....lazy.lazy import Lazy
from ....utils.space_utils import SpaceUtils
from .mobject_attribute import (
    InterpolateHandler,
    MobjectAttribute
)


_NPT = TypeVar("_NPT", bound=np.ndarray)


class ArrayAttribute(MobjectAttribute, Generic[_NPT]):
    __slots__ = ()

    def __init__(
        self,
        value: float | np.ndarray
    ) -> None:
        super().__init__()
        self._array_ = np.asarray(value, dtype=np.float64)

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _array_() -> _NPT:
        return NotImplemented

    @classmethod
    def _convert_input(
        cls,
        array_input: float | np.ndarray
    ) -> "ArrayAttribute[_NPT]":
        return ArrayAttribute(array_input)

    @classmethod
    def _interpolate(
        cls,
        array_0: "ArrayAttribute[_NPT]",
        array_1: "ArrayAttribute[_NPT]"
    ) -> "InterpolateHandler[ArrayAttribute[_NPT]]":
        return ArrayAttributeInterpolateHandler(array_0._array_, array_1._array_)


class ArrayAttributeInterpolateHandler(InterpolateHandler[ArrayAttribute[_NPT]]):
    __slots__ = (
        "_array_0",
        "_array_1"
    )

    def __init__(
        self,
        array_0: _NPT,
        array_1: _NPT
    ) -> None:
        super().__init__()
        self._array_0: _NPT = array_0
        self._array_1: _NPT = array_1

    def _interpolate(
        self,
        alpha: float
    ) -> ArrayAttribute[_NPT]:
        return ArrayAttribute(SpaceUtils.lerp(self._array_0, self._array_1, alpha))
