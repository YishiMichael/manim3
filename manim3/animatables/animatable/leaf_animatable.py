from abc import (
    ABC,
    abstractmethod
)
from typing import (
    Generic,
    TypeVar
)

from ...constants.custom_typing import (
    BoundaryT,
    NP_xf8
)
from ...lazy.lazy import Lazy
from .animatable import (
    Animatable,
    Updater
)
from .piecewiser import Piecewiser


_LeafAnimatableT = TypeVar("_LeafAnimatableT", bound="LeafAnimatable")


class LeafAnimatable(Animatable):
    __slots__ = ()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        assert not cls._animatable_descriptors

    @classmethod
    @abstractmethod
    def _interpolate(
        cls: type[_LeafAnimatableT],
        src_0: _LeafAnimatableT,
        src_1: _LeafAnimatableT
    ) -> "LeafAnimatableInterpolateInfo[_LeafAnimatableT]":
        pass

    @classmethod
    @abstractmethod
    def _split(
        cls: type[_LeafAnimatableT],
        #dst_tuple: tuple[_LeafAnimatableT, ...],
        src: _LeafAnimatableT,
        alphas: NP_xf8
    ) -> tuple[_LeafAnimatableT, ...]:
        pass

    @classmethod
    @abstractmethod
    def _concatenate(
        cls: type[_LeafAnimatableT],
        #dst: _LeafAnimatableT,
        src_tuple: tuple[_LeafAnimatableT, ...]
    ) -> _LeafAnimatableT:
        pass

    def _get_interpolate_updater(
        self: _LeafAnimatableT,
        src_0: _LeafAnimatableT,
        src_1: _LeafAnimatableT
    ) -> Updater:
        return super()._get_interpolate_updater(src_0, src_1).add(
            LeafAnimatableInterpolateUpdater(self, src_0, src_1)
        )

    def _get_piecewise_updater(
        self: _LeafAnimatableT,
        src: _LeafAnimatableT,
        piecewiser: Piecewiser
        #split_alphas: NP_xf8,
        #concatenate_indices: NP_xi4
    ) -> Updater:
        return super()._get_piecewise_updater(src, piecewiser).add(
            LeafAnimatablePiecewiseUpdater(self, src, piecewiser)
        )


class LeafAnimatableInterpolateInfo(ABC, Generic[_LeafAnimatableT]):
    __slots__ = ()

    def __init__(
        self,
        src_0: _LeafAnimatableT,
        src_1: _LeafAnimatableT
    ) -> None:
        super().__init__()

    @abstractmethod
    def interpolate(
        self,
        src: _LeafAnimatableT,
        alpha: float
    ) -> None:
        pass


class LeafAnimatableInterpolateUpdater(Generic[_LeafAnimatableT], Updater):
    __slots__ = ("_dst",)

    def __init__(
        self,
        dst: _LeafAnimatableT,
        src_0: _LeafAnimatableT,
        src_1: _LeafAnimatableT
    ) -> None:
        super().__init__()
        self._dst: _LeafAnimatableT = dst
        self._src_0_ = src_0.copy()
        self._src_1_ = src_1.copy()
        #self._positions_0_ = positions_0
        #self._positions_1_ = positions_1
        #self._edges_ = edges

    @Lazy.variable()
    @staticmethod
    def _src_0_() -> _LeafAnimatableT:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _src_1_() -> _LeafAnimatableT:
        return NotImplemented

    @Lazy.property()
    @staticmethod
    def _interpolate_info_(
        src_0: _LeafAnimatableT,
        src_1: _LeafAnimatableT
    ) -> LeafAnimatableInterpolateInfo[_LeafAnimatableT]:
        return type(src_0)._interpolate(src_0, src_1)

    def update(
        self,
        alpha: float
    ) -> None:
        super().update(alpha)
        self._interpolate_info_.interpolate(self._dst, alpha)

    def update_boundary(
        self,
        boundary: BoundaryT
    ) -> None:
        super().update_boundary(boundary)
        self._dst._copy_lazy_content(self._src_1_ if boundary else self._src_0_)


class LeafAnimatablePiecewiseUpdater(Generic[_LeafAnimatableT], Updater):
    __slots__ = (
        "_dst",
        "_src",
        "_piecewiser"
    )

    def __init__(
        self,
        dst: _LeafAnimatableT,
        src: _LeafAnimatableT,
        piecewise_func: Piecewiser
    ) -> None:
        super().__init__()
        self._dst: _LeafAnimatableT = dst
        self._src: _LeafAnimatableT = src
        self._piecewiser: Piecewiser = piecewise_func

    def update(
        self,
        alpha: float
    ) -> None:
        super().update(alpha)
        piecewise_data = self._piecewiser.piecewise(alpha)
        animatable_cls = type(self._dst)
        #pieces = tuple(animatable_cls() for _ in range(len(split_alphas) + 1))
        pieces = animatable_cls._split(self._src, piecewise_data.split_alphas)
        new_dst = animatable_cls._concatenate(tuple(pieces[index] for index in piecewise_data.concatenate_indices))
        self._dst._copy_lazy_content(new_dst)

    def update_boundary(
        self,
        boundary: BoundaryT
    ) -> None:
        super().update_boundary(boundary)
        self._dst._copy_lazy_content(self._src)
