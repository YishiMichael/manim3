from __future__ import annotations


from abc import (
    ABC,
    abstractmethod
)
from typing import (
    Iterator,
    Self,
    Unpack
)

from ...constants.custom_typing import (
    BoundaryT,
    NP_xf8
)
from .animatable import (
    Animatable,
    AnimatableAnimationBuilder,
    AnimateKwargs,
    Animation
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

    @property
    def _animate_cls(
        self: Self
    ) -> type[LeafAnimatableAnimationBuilder]:
        return LeafAnimatableAnimationBuilder

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
        #rate: Rate = Rates.linear(),
        #rewind: bool = False,
        #run_alpha: float = 1.0,
        #infinite: bool = False
    ) -> LeafAnimatableAnimationBuilder[Self]:
        return LeafAnimatableAnimationBuilder(self, **kwargs)


class LeafAnimatableAnimationBuilder[LeafAnimatableT: LeafAnimatable](AnimatableAnimationBuilder[LeafAnimatableT]):
    __slots__ = ()

    @classmethod
    def _iter_interpolate_animations(
        cls: type[Self],
        dst: LeafAnimatableT,
        src_0: LeafAnimatableT,
        src_1: LeafAnimatableT
    ) -> Iterator[Animation]:
        yield LeafAnimatableInterpolateAnimation(dst, src_0, src_1)
        yield from super()._iter_interpolate_animations(dst, src_0, src_1)

    @classmethod
    def _iter_piecewise_animations(
        cls: type[Self],
        dst: LeafAnimatableT,
        src: LeafAnimatableT,
        piecewiser: Piecewiser
        #split_alphas: NP_xf8,
        #concatenate_indices: NP_xi4
    ) -> Iterator[Animation]:
        yield LeafAnimatablePiecewiseAnimation(dst, src, piecewiser)
        yield from super()._iter_piecewise_animations(dst, src, piecewiser)


class LeafAnimatableInterpolateInfo[LeafAnimatableT: LeafAnimatable](ABC):
    __slots__ = ()

    @abstractmethod
    def interpolate(
        self: Self,
        dst: LeafAnimatableT,
        alpha: float
    ) -> None:
        pass


class LeafAnimatableInterpolateAnimation[LeafAnimatableT: LeafAnimatable](Animation):
    __slots__ = (
        "_dst",
        "_src_0",
        "_src_1",
        "_interpolate_info"
    )

    def __init__(
        self: Self,
        dst: LeafAnimatableT,
        src_0: LeafAnimatableT,
        src_1: LeafAnimatableT
    ) -> None:
        super().__init__()
        self._dst: LeafAnimatableT = dst
        self._src_0: LeafAnimatableT = src_0#.copy()
        self._src_1: LeafAnimatableT = src_1#.copy()
        self._interpolate_info: LeafAnimatableInterpolateInfo[LeafAnimatableT] | None = None
        #self._positions_0_ = positions_0
        #self._positions_1_ = positions_1
        #self._edges_ = edges

    #@Lazy.variable()
    #@staticmethod
    #def _src_0_() -> LeafAnimatableT:
    #    return NotImplemented

    #@Lazy.variable()
    #@staticmethod
    #def _src_1_() -> LeafAnimatableT:
    #    return NotImplemented

    #@Lazy.property()
    #@staticmethod
    #def _interpolate_info_(
    #    src_0: LeafAnimatableT,
    #    src_1: LeafAnimatableT
    #) -> LeafAnimatableInterpolateInfo[LeafAnimatableT]:
    #    return type(src_0)._interpolate(src_0, src_1)

    def update(
        self: Self,
        alpha: float
    ) -> None:
        super().update(alpha)
        if (interpolate_info := self._interpolate_info) is None:
            interpolate_info = type(self._dst)._interpolate(self._src_0, self._src_1)
            self._interpolate_info = interpolate_info
        interpolate_info.interpolate(self._dst, alpha)

    def update_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> None:
        super().update_boundary(boundary)
        self._dst._copy_lazy_content(self._src_1 if boundary else self._src_0)


class LeafAnimatablePiecewiseAnimation[LeafAnimatableT: LeafAnimatable](Animation):
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
        self.update(float(boundary))
        #self._dst._copy_lazy_content(self._src)
