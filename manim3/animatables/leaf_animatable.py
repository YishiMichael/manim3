from abc import abstractmethod
from typing import (
    Callable,
    TypeVar
)

from ..constants.custom_typing import (
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
    @classmethod
    def _interpolate(
        cls: type[_LeafAnimatableT],
        dst: _LeafAnimatableT,
        src_0: _LeafAnimatableT,
        src_1: _LeafAnimatableT
    ) -> Updater:
        pass

    @abstractmethod
    @classmethod
    def _split(
        cls: type[_LeafAnimatableT],
        dst_tuple: tuple[_LeafAnimatableT, ...],
        src: _LeafAnimatableT,
        alphas: NP_xf8
    ) -> None:
        pass

    @abstractmethod
    @classmethod
    def _concatenate(
        cls: type[_LeafAnimatableT],
        dst: _LeafAnimatableT,
        src_tuple: tuple[_LeafAnimatableT, ...]
    ) -> None:
        pass

    @classmethod
    def _get_interpolate_updater(
        cls: type[_LeafAnimatableT],
        dst: _LeafAnimatableT,
        src_0: _LeafAnimatableT,
        src_1: _LeafAnimatableT
    ) -> Updater:
        return super()._get_interpolate_updater(dst, src_0, src_1).add(
            cls._interpolate(dst, src_0, src_1)
        )

    @classmethod
    def _get_piecewise_updater(
        cls: type[_LeafAnimatableT],
        dst: _LeafAnimatableT,
        src: _LeafAnimatableT,
        piecewise_func: Callable[[float], tuple[NP_xf8, NP_xi4]]
        #split_alphas: NP_xf8,
        #concatenate_indices: NP_xi4
    ) -> Updater:
        return super()._get_piecewise_updater(dst, src, piecewise_func).add(
            LeafAnimatablePiecewiseUpdater(dst, src, piecewise_func)
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
        pieces = tuple(animatable_cls() for _ in range(len(split_alphas) + 1))
        animatable_cls._split(pieces, self._src, split_alphas)
        animatable_cls._concatenate(self._dst, tuple(pieces[index] for index in concatenate_indices))

    def initial_update(self) -> None:
        super().initial_update()
        self.update(0.0)

    def final_update(self) -> None:
        super().final_update()
        self.update(1.0)
