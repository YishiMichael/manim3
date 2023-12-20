from __future__ import annotations


from abc import abstractmethod
from typing import (
    Iterator,
    Self,
    Unpack
)

from ...constants.custom_typing import NP_xf8
from ...lazy.lazy import Lazy
from ...lazy.lazy_object import LazyObject
from .action import (
    Action,
    DescriptiveAction,
    DescriptorParameters
)
from .animation import (
    AnimateKwargs,
    Animation,
    AnimationsTimeline
)
from .piecewiser import Piecewiser


class Animatable(LazyObject):
    __slots__ = ()

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
    ) -> AnimatableTimeline[Self]:
        return AnimatableTimeline(self, **kwargs)

    @DescriptiveAction.descriptive_register(DescriptorParameters)
    @classmethod
    def interpolate(
        cls: type[Self],
        dst: Self,
        src_0: Self,
        src_1: Self
    ) -> Iterator[Animation]:
        for descriptor, _ in cls.interpolate.iter_descriptor_items():
            if not all(
                descriptor in animatable._lazy_descriptors
                for animatable in (dst, src_0, src_1)
            ):
                continue
            for dst_element, src_0_element, src_1_element in zip(
                descriptor.get_elements(dst),
                descriptor.get_elements(src_0),
                descriptor.get_elements(src_1),
                strict=True
            ):
                assert isinstance(dst_element, Animatable)
                yield from type(dst_element).interpolate.iter_animations(
                    dst=dst_element,
                    src_0=src_0_element,
                    src_1=src_1_element
                )

    @DescriptiveAction.descriptive_register(DescriptorParameters)
    @classmethod
    def piecewise(
        cls: type[Self],
        dst: Self,
        src: Self,
        piecewiser: Piecewiser
    ) -> Iterator[Animation]:
        for descriptor, _ in cls.piecewise.iter_descriptor_items():
            if not all(
                descriptor in animatable._lazy_descriptors
                for animatable in (dst, src)
            ):
                continue
            for dst_element, src_element in zip(
                descriptor.get_elements(dst),
                descriptor.get_elements(src),
                strict=True
            ):
                assert isinstance(dst_element, Animatable)
                yield from type(dst_element).piecewise.iter_animations(
                    dst=dst_element,
                    src=src_element,
                    piecewiser=piecewiser
                )

    @Action.register()
    @classmethod
    def transform(
        cls: type[Self],
        dst: Self,
        src: Self
    ) -> Iterator[Animation]:
        yield from cls.interpolate.iter_animations(
            dst=dst,
            src_0=dst.copy(),
            src_1=src
        )


class AnimatableTimeline[AnimatableT: Animatable](AnimationsTimeline):
    __slots__ = ("_dst",)

    def __init__(
        self: Self,
        dst: AnimatableT,
        **kwargs: Unpack[AnimateKwargs]
    ) -> None:
        super().__init__(**kwargs)
        self._dst: AnimatableT = dst

    interpolate = Animatable.interpolate
    piecewise = Animatable.piecewise
    transform = Animatable.transform


class AnimatableInterpolateAnimation[AnimatableT: Animatable](Animation):
    __slots__ = (
        "_dst",
        "_interpolate_info"
    )

    def __init__(
        self: Self,
        dst: AnimatableT,
        src_0: AnimatableT,
        src_1: AnimatableT
    ) -> None:
        super().__init__()
        self._dst: AnimatableT = dst
        self._src_0_ = src_0.copy()
        self._src_1_ = src_1.copy()

    @Lazy.variable()
    @staticmethod
    def _src_0_() -> AnimatableT:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _src_1_() -> AnimatableT:
        return NotImplemented

    @abstractmethod
    def interpolate(
        self: Self,
        dst: AnimatableT,
        alpha: float
    ) -> None:
        pass

    def update(
        self: Self,
        alpha: float
    ) -> None:
        super().update(alpha)
        self.interpolate(self._dst, alpha)


class AnimatablePiecewiseAnimation[AnimatableT: Animatable](Animation):
    __slots__ = (
        "_dst",
        "_src",
        "_piecewiser"
    )

    def __init__(
        self: Self,
        dst: AnimatableT,
        src: AnimatableT,
        piecewise_func: Piecewiser
    ) -> None:
        super().__init__()
        self._dst: AnimatableT = dst
        self._src: AnimatableT = src
        self._piecewiser: Piecewiser = piecewise_func

    @classmethod
    @abstractmethod
    def split(
        cls: type[Self],
        dsts: tuple[AnimatableT, ...],
        src: AnimatableT,
        alphas: NP_xf8
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def concatenate(
        cls: type[Self],
        dst: AnimatableT,
        srcs: tuple[AnimatableT, ...]
    ) -> None:
        pass

    def update(
        self: Self,
        alpha: float
    ) -> None:
        super().update(alpha)
        cls = type(self)
        piecewise_info = self._piecewiser.piecewise(alpha)
        dst = self._dst
        animatable_cls = type(dst)
        pieces = tuple(animatable_cls() for _ in range(len(piecewise_info.split_alphas) + 1))
        cls.split(pieces, self._src, piecewise_info.split_alphas)
        cls.concatenate(dst, tuple(pieces[index] for index in piecewise_info.concatenate_indices))
