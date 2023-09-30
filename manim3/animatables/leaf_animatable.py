from abc import abstractmethod
from typing import (
    Callable,
    TypeVar
)

from ..constants.custom_typing import (
    BoundaryT,
    NP_xf8,
    NP_xi4
)
from .animatable import (
    Animatable,
    Updater
)


_LeafAnimatableT = TypeVar("_LeafAnimatableT", bound="LeafAnimatable")


class LeafAnimatable(Animatable):
    __slots__ = ()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        assert not cls._animatable_descriptors

    @abstractmethod
    def _interpolate(
        self: _LeafAnimatableT,
        src_0: _LeafAnimatableT,
        src_1: _LeafAnimatableT
    ) -> Updater:
        pass

    @abstractmethod
    @classmethod
    def _split(
        cls,
        #dst_tuple: tuple[_LeafAnimatableT, ...],
        src: _LeafAnimatableT,
        alphas: NP_xf8
    ) -> tuple[_LeafAnimatableT, ...]:
        pass

    @abstractmethod
    @classmethod
    def _concatenate(
        cls,
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
            self._interpolate(src_0, src_1)
        )

    def _get_piecewise_updater(
        self: _LeafAnimatableT,
        src: _LeafAnimatableT,
        piecewise_func: Callable[[float], tuple[NP_xf8, NP_xi4]]
        #split_alphas: NP_xf8,
        #concatenate_indices: NP_xi4
    ) -> Updater:
        return super()._get_piecewise_updater(src, piecewise_func).add(
            LeafAnimatablePiecewiseUpdater(self, src, piecewise_func)
        )


class LeafAnimatablePiecewiseUpdater(Updater):
    __slots__ = ()

    def __init__(
        self,
        dst: _LeafAnimatableT,
        src: _LeafAnimatableT,
        piecewise_func: Callable[[float], tuple[NP_xf8, NP_xi4]]
    ) -> None:
        super().__init__()
        self._dst: _LeafAnimatableT = dst
        self._src: _LeafAnimatableT = src
        self._piecewise_func: Callable[[float], tuple[NP_xf8, NP_xi4]] = piecewise_func

    def update(
        self,
        alpha: float
    ) -> None:
        super().update(alpha)
        split_alphas, concatenate_indices = self._piecewise_func(alpha)
        animatable_cls = type(self._dst)
        #pieces = tuple(animatable_cls() for _ in range(len(split_alphas) + 1))
        pieces = animatable_cls._split(self._src, split_alphas)
        new_dst = animatable_cls._concatenate(tuple(pieces[index] for index in concatenate_indices))
        self._dst._copy_lazy_content(new_dst)

    def update_boundary(
        self,
        boundary: BoundaryT
    ) -> None:
        super().update_boundary(boundary)
        self._dst._copy_lazy_content(self._src)
