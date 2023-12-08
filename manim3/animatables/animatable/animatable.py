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
from .actions import (
    Action,
    Actions
)
from .animation import (
    AnimateKwargs,
    Animation,
    AnimationsTimeline
)
from .piecewiser import Piecewiser


class AnimatableActions(Actions):
    __slots__ = ()

    @Action.register()
    @classmethod
    def interpolate(
        cls: type[Self],
        dst: Animatable,
        src_0: Animatable,
        src_1: Animatable
    ) -> Iterator[Animation]:
        for descriptor in cls.interpolate._descriptor_dict:
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
                yield from type(dst_element).interpolate._action(
                    dst=dst_element,
                    src_0=src_0_element,
                    src_1=src_1_element
                )

    @Action.register()
    @classmethod
    def piecewise(
        cls: type[Self],
        dst: Animatable,
        src: Animatable,
        piecewiser: Piecewiser
    ) -> Iterator[Animation]:
        for descriptor in cls.piecewise._descriptor_dict:
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
                yield from type(dst_element).piecewise._action(
                    dst=dst_element,
                    src=src_element,
                    piecewiser=piecewiser
                )

    @Action.register()
    @classmethod
    def transform(
        cls: type[Self],
        dst: Animatable,
        src: Animatable
    ) -> Iterator[Animation]:
        yield from cls.interpolate(
            dst=dst,
            src_0=dst.copy(),
            src_1=src
        )


class Animatable(LazyObject):
    __slots__ = ()

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
    ) -> DynamicAnimatable[Self]:
        return DynamicAnimatable(self, **kwargs)

    interpolate = AnimatableActions.interpolate.build_action_descriptor()
    piecewise = AnimatableActions.piecewise.build_action_descriptor()
    transform = AnimatableActions.transform.build_action_descriptor()


class DynamicAnimatable[AnimatableT: Animatable](AnimationsTimeline):
    __slots__ = ("_dst",)

    def __init__(
        self: Self,
        dst: AnimatableT,
        **kwargs: Unpack[AnimateKwargs]
    ) -> None:
        super().__init__(**kwargs)
        self._dst: AnimatableT = dst

    interpolate = AnimatableActions.interpolate.build_dynamic_action_descriptor()
    piecewise = AnimatableActions.piecewise.build_dynamic_action_descriptor()
    transform = AnimatableActions.transform.build_dynamic_action_descriptor()


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
        piecewise_data = self._piecewiser.piecewise(alpha)
        dst = self._dst
        animatable_cls = type(dst)
        pieces = tuple(animatable_cls() for _ in range(len(piecewise_data.split_alphas) + 1))
        cls.split(pieces, self._src, piecewise_data.split_alphas)
        cls.concatenate(dst, tuple(pieces[index] for index in piecewise_data.concatenate_indices))
