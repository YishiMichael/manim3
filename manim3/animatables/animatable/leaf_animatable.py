from __future__ import annotations


from abc import (
    ABC,
    abstractmethod
)
from typing import Self

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


class LeafAnimatable(Animatable):
    __slots__ = ()

    def __init_subclass__(
        cls: type[Self]
    ) -> None:
        super().__init_subclass__()
        assert not cls._animatable_descriptors

    @classmethod
    @abstractmethod
    def _interpolate(
        cls: type[Self],
        src_0: Self,
        src_1: Self
    ) -> LeafAnimatableInterpolateInfo[Self]:
        pass

    @classmethod
    @abstractmethod
    def _split(
        cls: type[Self],
        src: Self,
        alphas: NP_xf8
    ) -> tuple[Self, ...]:
        pass

    @classmethod
    @abstractmethod
    def _concatenate(
        cls: type[Self],
        #dst: Self,
        src_tuple: tuple[Self, ...]
    ) -> Self:
        pass

    def _get_interpolate_updater(
        self: Self,
        src_0: Self,
        src_1: Self
    ) -> Updater:
        return super()._get_interpolate_updater(src_0, src_1).add(
            LeafAnimatableInterpolateUpdater(self, src_0, src_1)
        )

    def _get_piecewise_updater(
        self: Self,
        src: Self,
        piecewiser: Piecewiser
        #split_alphas: NP_xf8,
        #concatenate_indices: NP_xi4
    ) -> Updater:
        return super()._get_piecewise_updater(src, piecewiser).add(
            LeafAnimatablePiecewiseUpdater(self, src, piecewiser)
        )


class LeafAnimatableInterpolateInfo[LeafAnimatableT: LeafAnimatable](ABC):
    __slots__ = ()

    def __init__(
        self: Self,
        src_0: LeafAnimatableT,
        src_1: LeafAnimatableT
    ) -> None:
        super().__init__()

    @abstractmethod
    def interpolate(
        self: Self,
        src: LeafAnimatableT,
        alpha: float
    ) -> None:
        pass


class LeafAnimatableInterpolateUpdater[LeafAnimatableT: LeafAnimatable](Updater):
    __slots__ = ("_dst",)

    def __init__(
        self: Self,
        dst: LeafAnimatableT,
        src_0: LeafAnimatableT,
        src_1: LeafAnimatableT
    ) -> None:
        super().__init__()
        self._dst: LeafAnimatableT = dst
        self._src_0_ = src_0.copy()
        self._src_1_ = src_1.copy()
        #self._positions_0_ = positions_0
        #self._positions_1_ = positions_1
        #self._edges_ = edges

    @Lazy.variable()
    @staticmethod
    def _src_0_() -> LeafAnimatableT:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _src_1_() -> LeafAnimatableT:
        return NotImplemented

    @Lazy.property()
    @staticmethod
    def _interpolate_info_(
        src_0: LeafAnimatableT,
        src_1: LeafAnimatableT
    ) -> LeafAnimatableInterpolateInfo[LeafAnimatableT]:
        return type(src_0)._interpolate(src_0, src_1)

    def update(
        self: Self,
        alpha: float
    ) -> None:
        super().update(alpha)
        self._interpolate_info_.interpolate(self._dst, alpha)

    def update_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> None:
        super().update_boundary(boundary)
        self._dst._copy_lazy_content(self._src_1_ if boundary else self._src_0_)


class LeafAnimatablePiecewiseUpdater[LeafAnimatableT: LeafAnimatable](Updater):
    __slots__ = (
        "_dst",
        "_src",
        "_piecewiser"
    )

    def __init__(
        self: Self,
        dst: LeafAnimatableT,
        src: LeafAnimatableT,
        piecewise_func: Piecewiser
    ) -> None:
        super().__init__()
        self._dst: LeafAnimatableT = dst
        self._src: LeafAnimatableT = src
        self._piecewiser: Piecewiser = piecewise_func

    def update(
        self: Self,
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
        self: Self,
        boundary: BoundaryT
    ) -> None:
        super().update_boundary(boundary)
        self._dst._copy_lazy_content(self._src)
