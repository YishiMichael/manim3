from abc import (
    ABC,
    abstractmethod
)
from typing import (
    ClassVar,
    Generic,
    TypeVar
)

from ....constants.custom_typing import NP_xf8
from ....lazy.lazy import LazyObject


_MobjectAttributeT = TypeVar("_MobjectAttributeT", bound="MobjectAttribute")


class MobjectAttribute(LazyObject):
    __slots__ = ()

    _interpolate_implemented: ClassVar[bool] = False
    _split_implemented: ClassVar[bool] = False
    _concatenate_implemented: ClassVar[bool] = False

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        base_cls = cls.__base__
        assert issubclass(base_cls, MobjectAttribute)
        cls._interpolate_implemented = base_cls._interpolate_implemented or "_interpolate" in cls.__dict__
        cls._split_implemented = base_cls._split_implemented or "_split" in cls.__dict__
        cls._concatenate_implemented = base_cls._concatenate_implemented or "_concatenate" in cls.__dict__

    @classmethod
    def _convert_input(
        cls: type[_MobjectAttributeT],
        attribute_input: _MobjectAttributeT
    ) -> _MobjectAttributeT:
        return attribute_input

    @classmethod
    def _interpolate(
        cls: type[_MobjectAttributeT],
        attribute_0: _MobjectAttributeT,
        attribute_1: _MobjectAttributeT
    ) -> "InterpolateHandler[_MobjectAttributeT]":
        return NotImplemented

    @classmethod
    def _split(
        cls: type[_MobjectAttributeT],
        attribute: _MobjectAttributeT,
        alphas: NP_xf8
    ) -> list[_MobjectAttributeT]:
        return NotImplemented

    @classmethod
    def _concatenate(
        cls: type[_MobjectAttributeT],
        attribute_list: list[_MobjectAttributeT]
    ) -> _MobjectAttributeT:
        return NotImplemented


class InterpolateHandler(ABC, Generic[_MobjectAttributeT]):
    __slots__ = ()

    @abstractmethod
    def _interpolate(
        self,
        alpha: float
    ) -> _MobjectAttributeT:
        pass
