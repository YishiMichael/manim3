from __future__ import annotations
from abc import abstractmethod


#from abc import ABC
from typing import (
    #TYPE_CHECKING,
    #ClassVar,
    #Iterable,
    Iterator,
    Self,
    #TypedDict,
    Unpack
)

from ...lazy.lazy_descriptor import LazyDescriptor

#from ...timelines.timeline.rate import Rate
#from ..timelines.timeline.rates.linear import Linear
#from ..timelines.timeline.rates.rate import Rate
#from ...constants.custom_typing import (
#    BoundaryT,
#    ColorT
#)
from ...constants.custom_typing import (
    BoundaryT,
    NP_xf8
)
#from ...lazy.lazy_descriptor import LazyDescriptor
from ...lazy.lazy_object import LazyObject
#from .action_metadata import ActionDescriptor
from .actions import (
    ActionMeta,
    Actions
)
from .animation import (
    AnimateKwargs,
    Animation,
    AnimationsTimeline
)
from .piecewiser import Piecewiser


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class AnimateInfo:
#    rate: Rate
#    rewind: bool
#    infinite: bool


class AnimatableActions(Actions):
    __slots__ = ()

    @ActionMeta.register
    @classmethod
    def interpolate(
        cls: type[Self],
        dst: Animatable,
        src_0: Animatable,
        src_1: Animatable
    ) -> Iterator[Animation]:
        for descriptor in type(dst)._lazy_descriptors:
            #assert issubclass(descriptor._element_type, Animatable)
            #dst_elements = descriptor._get_elements(dst)
            #src_0_elements = descriptor._get_elements(src_0)
            #src_1_elements = descriptor._get_elements(src_1)
            #if not issubclass(element_type, Animatable) and src_0_elements != src_1_elements:
            #    raise ValueError(f"Uninterpolable variables of `{descriptor._name}` don't match")
            for dst_element, src_0_element, src_1_element in zip(
                descriptor.get_elements(dst),
                descriptor.get_elements(src_0),
                descriptor.get_elements(src_1),
                strict=True
            ):
                if not isinstance(dst_element, Animatable):
                    continue
                yield from type(dst_element).interpolate(
                    dst=dst_element,
                    src_0=src_0_element,
                    src_1=src_1_element
                )

    @ActionMeta.register
    @classmethod
    def piecewise(
        cls: type[Self],
        dst: Animatable,
        src: Animatable,
        piecewiser: Piecewiser
        #split_alphas: NP_xf8,
        #concatenate_indices: NP_xi4
    ) -> Iterator[Animation]:
        for descriptor in type(dst)._lazy_descriptors:
            #assert issubclass(descriptor._element_type, Animatable)
            #dst_elements = descriptor._get_elements(dst)
            #src_elements = descriptor._get_elements(src)
            for dst_element, src_element in zip(
                descriptor.get_elements(dst),
                descriptor.get_elements(src),
                strict=True
            ):
                if not isinstance(dst_element, Animatable):
                    continue
                yield from type(dst_element).piecewise(
                    dst=dst_element,
                    src=src_element,
                    piecewiser=piecewiser
                )

    @ActionMeta.register
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


class Animatable(AnimatableActions, LazyObject):
    __slots__ = ()

    #_animatable_descriptors: ClassVar[dict[str, LazyDescriptor]] = {}

    #def __init_subclass__(
    #    cls: type[Self]
    #) -> None:
    #    super().__init_subclass__()
    #    print(cls.__name__, [descriptor._name for descriptor in cls.__dict__.values() if isinstance(descriptor, LazyDescriptor)])
    #    print()
    #    cls._animatable_descriptors = {
    #        name: descriptor
    #        for name, descriptor in cls._lazy_descriptors.items()
    #        if descriptor._is_variable
    #        and descriptor._name not in (
    #            "_siblings_",
    #            "_camera_",
    #            "_lighting_"
    #        )  # TODO
    #        #and descriptor._element_type is not None
    #        #and issubclass(descriptor._element_type, Animatable)
    #        #and descriptor._element_type is not cls
    #    }

    #def __init__(
    #    self: Self
    #) -> None:
    #    super().__init__()
    #    self._animate_info: AnimateInfo | None = None
    #    self._animations: list[Animation] = []
    #    #self._reset_animations()

    #@classmethod
    #def _stack_animations(
    #    cls,
    #    method: "Callable[Concatenate[_T, _P], Iterator[Animation]]"
    #) -> Callable[Concatenate[_T, _P], _T]:

    #    def result(
    #        self: _T,
    #        *args: _P.args,
    #        **kwargs: _P.kwargs
    #    ) -> _T:
    #        for animation in method(self, *args, **kwargs):
    #            self._animations.append(animation)
    #            animation.update(1.0)
    #        return self

    #    return result

    #@classmethod
    #@property
    #def _animatable_descriptors(cls) -> Iterator[LazyDescriptor]:
    #    for descriptor in cls._lazy_descriptors.values():
    #        if descriptor._is_variable and descriptor._element_type is not cls:
    #            yield descriptor

    #def _reset_animations(
    #    self: Self
    #) -> None:
    #    self._animations.clear()

    @classmethod
    def _convert_input(
        cls: type[Self],
        animatable_input: Self
    ) -> Self:
        return animatable_input

    #@classmethod
    #def _iter_elements_from_input(
    #    cls: type[Self],
    #    descriptor: LazyDescriptor,
    #    animatable_input: Any
    #) -> Iterator:
    #    if descriptor._is_plural:
    #        for animatable_input_element in animatable_input:
    #            yield cls._convert_input(animatable_input_element)
    #    else:
    #        yield cls._convert_input(animatable_input)

    #def _iter_siblings(
    #    self: Self,
    #    *,
    #    broadcast: bool = True
    #) -> Iterator[Animatable]:
    #    yield self

    #@classmethod
    #def _split(
    #    cls: type[_AnimatableT],
    #    dst_tuple: tuple[_AnimatableT, ...],
    #    src: _AnimatableT,
    #    alphas: NP_xf8
    #) -> None:
    #    for descriptor in cls._animatable_descriptors.values():
    #        assert issubclass(element_type := descriptor._element_type, Animatable)
    #        for dst_element_tuple, src_element in zip(
    #            tuple(descriptor._get_elements(dst) for dst in dst_tuple),
    #            descriptor._get_elements(src),
    #            strict=True
    #        ):
    #            element_type._split(dst_element_tuple, src_element, alphas)

    #@classmethod
    #def _concatenate(
    #    cls: type[_AnimatableT],
    #    dst: _AnimatableT,
    #    srcs: tuple[_AnimatableT, ...]
    #) -> None:
    #    for descriptor in cls._animatable_descriptors.values():
    #        assert issubclass(element_type := descriptor._element_type, Animatable)
    #        for dst_element, src_element_tuple in zip(
    #            descriptor._get_elements(dst),
    #            tuple(descriptor._get_elements(src) for src in srcs),
    #            strict=True
    #        ):
    #            element_type._concatenate(dst_element, src_element_tuple)

    #def _get_interpolate_animation(
    #    self: _AnimatableT,
    #    animatable_0: _AnimatableT,
    #    animatable_1: _AnimatableT
    #) -> Animation:
    #    raise NotImplementedError

    #def interpolate(
    #    self,
    #    animatable_0: _AnimatableT,
    #    animatable_1: _AnimatableT
    #):
    #    self._stack_animation(type(self)._interpolate(
    #        self, animatable_0, animatable_1
    #    ))
    #    return self

    #def submit_timeline(
    #    self: Self
    #) -> Timeline:
    #    #assert not infinite or not rewind
    #    #assert (saved_state := self._saved_state) is not None
    #    #self._copy_lazy_content(self, saved_state)
    #    assert (animate_info := self._animate_info) is not None, "Cannot submit timeline"
    #    rate = animate_info.rate
    #    if animate_info.rewind:
    #        rate = Rates.compose(rate, Rates.rewind())
    #    run_alpha = float("inf") if animate_info.infinite else 1.0
    #    timeline = AnimationsTimeline(
    #        animations=tuple(self._animations),
    #        rate=rate,
    #        run_alpha=run_alpha
    #    )
    #    #timeline = AnimationTimeline(
    #    #    #instance=self,
    #    #    animation=self._animation,
    #    #    rate=rate,
    #    #    run_alpha=float("inf") if infinite else 1.0
    #    #    #infinite=infinite,
    #    #    #run_alpha=float("inf") if infinite else 1.0,
    #    #    #rewind=rewind
    #    #)
    #    self._animate_info = None
    #    #self._animation = Animation()
    #    self._animations.clear()
    #    return timeline

    #@property
    #def _animate_cls(
    #    self: Self
    #) -> type[AnimatableAnimationBuilder]:
    #    return AnimatableAnimationBuilder

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
        #rate: Rate = Rates.linear(),
        #rewind: bool = False,
        #run_alpha: float = 1.0,
        #infinite: bool = False
    ) -> DynamicAnimatable[Self]:
        return DynamicAnimatable(self, **kwargs)
        #assert self._animate_info is None, "Existing timeline has not been submitted"
        #self._animate_info = AnimateInfo(
        #    rate=rate,
        #    rewind=rewind,
        #    infinite=infinite
        #)
        #self._animations.clear()
        #return self

    #def interpolate(
    #    self: Self,
    #    animatable_0: Self,
    #    animatable_1: Self,
    #    *,
    #    broadcast: bool = True,
    #    alpha: float = 1.0
    #) -> Self:
    #    self.animate().interpolate(animatable_0, animatable_1, broadcast=broadcast).update(alpha)
    #    return self

    #def piecewise(
    #    self: Self,
    #    animatable: Self,
    #    piecewiser: Piecewiser,
    #    *,
    #    broadcast: bool = True,
    #    alpha: float = 1.0
    #) -> Self:
    #    self.animate().piecewise(animatable, piecewiser, broadcast=broadcast).update(alpha)
    #    return self

    #def set(
    #    self: Self,
    #    *,
    #    broadcast: bool = True,
    #    **kwargs: Unpack[SetKwargs]
    #) -> Self:
    #    #for dst_sibling in self._iter_siblings(broadcast=broadcast):
    #    #    for name, animatable_input in kwargs.items():
    #    #        if (descriptor := type(dst_sibling)._animatable_descriptors.get(f"_{name}_")) is None:
    #    #            continue
    #    #        assert (element_type := descriptor._element_type) is not None and issubclass(element_type, Animatable)
    #    #        target_elements = tuple(element_type._iter_elements_from_input(descriptor, animatable_input))
    #    #        #if descriptor._is_plural:
    #    #        #    target_elements = tuple(
    #    #        #        element_type._convert_input(animatable_input_element)
    #    #        #        for animatable_input_element in animatable_input
    #    #        #    )
    #    #        #else:
    #    #        #    target_elements = (element_type._convert_input(animatable_input),)
    #    #        descriptor._set_elements(dst_sibling, target_elements)
    #    self.animate().set(broadcast=broadcast, **kwargs).update_boundary(1)
    #    #self._stack_animations(self._iter_set_animations(self._animatable, **kwargs))
    #    return self

    #def transform(
    #    self: Self,
    #    animatable: Self,
    #    *,
    #    broadcast: bool = True
    #) -> Self:
    #    self.animate().transform(animatable, broadcast=broadcast).update_boundary(1)
    #    #self.interpolate(self._animatable.copy(), animatable)
    #    return self

    #def static_interpolate(
    #    self: Self,
    #    animatable_0: Self,
    #    animatable_1: Self,
    #    alpha: float
    #) -> Self:
    #    for animation in self._iter_interpolate_animations(animatable_0, animatable_1):
    #        animation.update(alpha)
    #    return self

    #def static_piecewise(
    #    self: Self,
    #    animatable: Self,
    #    piecewiser: Piecewiser,
    #    alpha: float
    #) -> Self:
    #    for animation in self._iter_piecewise_animations(animatable, piecewiser):
    #        animation.update(alpha)
    #    return self


class DynamicAnimatable[AnimatableT: Animatable](AnimatableActions, AnimationsTimeline):
    __slots__ = ("_dst",)

    def __init__(
        self: Self,
        dst: AnimatableT,
        **kwargs: Unpack[AnimateKwargs]
    ) -> None:
        super().__init__(**kwargs)
        self._dst: AnimatableT = dst


#class AnimatableAnimationsTimeline[AnimatableT: Animatable](AnimatableActions, DynamicAnimatable[AnimatableT]):
#    __slots__ = ()

    #def __init__(
    #    self: Self,
    #    animatable: AnimatableT,
    #    #instance: _T,
    #    #animation: Animation
    #    #animation: Animation,
    #    **kwargs: Unpack[AnimateKwargs]
    #    #rate: Rate,
    #    #rewind: bool,
    #    #run_alpha: float = 1.0,
    #    #infinite: bool
    #) -> None:
    #    super().__init__(**kwargs)
    #    self._animatable: AnimatableT = animatable

    #@classmethod
    #def _iter_interpolate_animations(
    #    cls: type[Self],
    #    dst: AnimatableT,
    #    src_0: AnimatableT,
    #    src_1: AnimatableT
    #) -> Iterator[Animation]:
    #    for descriptor in type(dst)._animatable_descriptors.values():
    #        #assert issubclass(descriptor._element_type, Animatable)
    #        #dst_elements = descriptor._get_elements(dst)
    #        #src_0_elements = descriptor._get_elements(src_0)
    #        #src_1_elements = descriptor._get_elements(src_1)
    #        #if not issubclass(element_type, Animatable) and src_0_elements != src_1_elements:
    #        #    raise ValueError(f"Uninterpolable variables of `{descriptor._name}` don't match")
    #        for dst_element, src_0_element, src_1_element in zip(
    #            descriptor._get_elements(dst),
    #            descriptor._get_elements(src_0),
    #            descriptor._get_elements(src_1),
    #            strict=True
    #        ):
    #            yield from dst_element._animate_cls._iter_interpolate_animations(
    #                dst=dst_element,
    #                src_0=src_0_element,
    #                src_1=src_1_element
    #            )

        #if broadcast:
        #    for dst_sibling, src_0_sibling, src_1_sibling in zip(
        #        dst._iter_siblings(), src_0._iter_siblings(), src_1._iter_siblings()
        #    ):
        #        cls._iter_interpolate_animations(dst_sibling, src_0_sibling, src_1_sibling, broadcast=False)

    #@classmethod
    #def _iter_piecewise_animations(
    #    cls: type[Self],
    #    dst: AnimatableT,
    #    src: AnimatableT,
    #    piecewiser: Piecewiser
    #    #split_alphas: NP_xf8,
    #    #concatenate_indices: NP_xi4
    #) -> Iterator[Animation]:
    #    for descriptor in type(dst)._animatable_descriptors.values():
    #        #assert issubclass(descriptor._element_type, Animatable)
    #        #dst_elements = descriptor._get_elements(dst)
    #        #src_elements = descriptor._get_elements(src)
    #        for dst_element, src_element in zip(
    #            descriptor._get_elements(dst),
    #            descriptor._get_elements(src),
    #            strict=True
    #        ):
    #            yield from dst_element._animate_cls._iter_piecewise_animations(
    #                dst=dst_element,
    #                src=src_element,
    #                piecewiser=piecewiser
    #            )

        #if broadcast:
        #    for dst_sibling, src_sibling in zip(
        #        dst._iter_siblings(), src._iter_siblings()
        #    ):
        #        cls._iter_piecewise_animations(dst_sibling, src_sibling, piecewiser, broadcast=False)
        #pieces = tuple(cls() for _ in range(len(split_alphas) + 1))
        #cls._split(pieces, src, split_alphas)
        #cls._concatenate(dst, tuple(pieces[index] for index in concatenate_indices))

    #@classmethod
    #def _iter_set_animations(
    #    cls: type[Self],
    #    dst: AnimatableT,
    #    **kwargs: Unpack[SetKwargs]
    #) -> Iterator[Animation]:
    #    for name, animatable_input in kwargs.items():
    #        if (descriptor := type(dst)._animatable_descriptors.get(f"_{name}_")) is not None:
    #            yield AnimatableSetAnimation(dst, descriptor, animatable_input)
    #        #assert (element_type := descriptor._element_type) is not None and issubclass(element_type, Animatable)
    #        #source_elements = descriptor._get_elements(dst)
    #        #target_elements = tuple(element_type._iter_elements_from_input(descriptor, animatable_input))
    #        ##if descriptor._is_plural:
    #        ##    target_elements = tuple(
    #        ##        element_type._convert_input(animatable_input_element)
    #        ##        for animatable_input_element in animatable_input
    #        ##    )
    #        ##else:
    #        ##    target_elements = (element_type._convert_input(animatable_input),)
    #        ##descriptor.__set__(self, target_animatable)
    #        ##if self._saved_state is not None:
    #        #for source_element, target_element in zip(source_elements, target_elements, strict=True):
    #        #    yield from cls._iter_interpolate_animations(
    #        #        dst=source_element,
    #        #        src_0=source_element.copy(),
    #        #        src_1=target_element.copy()
    #        #    )
    #    #if broadcast:
    #    #    for dst_sibling in dst._iter_siblings():
    #    #        cls._iter_set_animations(dst_sibling, broadcast=False, **kwargs)

    #def interpolate(
    #    self: Self,
    #    animatable_0: AnimatableT,
    #    animatable_1: AnimatableT,
    #    *,
    #    broadcast: bool = True
    #) -> Self:
    #    self._stack_animations(itertools.chain.from_iterable(
    #        dst_sibling._animate_cls._iter_interpolate_animations(
    #            dst=dst_sibling,
    #            src_0=src_0_sibling,
    #            src_1=src_1_sibling
    #        )
    #        for dst_sibling, src_0_sibling, src_1_sibling in zip(
    #            self._animatable._iter_siblings(broadcast=broadcast),
    #            animatable_0._iter_siblings(broadcast=broadcast),
    #            animatable_1._iter_siblings(broadcast=broadcast)
    #        )
    #    ))
    #    return self

    #def piecewise(
    #    self: Self,
    #    animatable: AnimatableT,
    #    piecewiser: Piecewiser,
    #    *,
    #    broadcast: bool = True
    #) -> Self:
    #    self._stack_animations(itertools.chain.from_iterable(
    #        dst_sibling._animate_cls._iter_piecewise_animations(
    #            dst=dst_sibling,
    #            src=src_sibling,
    #            piecewiser=piecewiser
    #        )
    #        for dst_sibling, src_sibling in zip(
    #            self._animatable._iter_siblings(broadcast=broadcast),
    #            animatable._iter_siblings(broadcast=broadcast)
    #        )
    #    ))
    #    return self

    #def set(
    #    self: Self,
    #    *,
    #    broadcast: bool = True,
    #    **kwargs: Unpack[SetKwargs]
    #) -> Self:
    #    self._stack_animations(itertools.chain.from_iterable(
    #        dst_sibling._animate_cls._iter_set_animations(
    #            dst=dst_sibling,
    #            **kwargs
    #        )
    #        for dst_sibling in self._animatable._iter_siblings(broadcast=broadcast)
    #    ))
    #    return self

    #def transform(
    #    self: Self,
    #    animatable: AnimatableT,
    #    *,
    #    broadcast: bool = True
    #) -> Self:
    #    self.interpolate(
    #        animatable_0=self._animatable.copy(),
    #        animatable_1=animatable,
    #        broadcast=broadcast
    #    )
    #    return self


#class Animation(LazyObject):
#    __slots__ = ()

#    #def __init__(
#    #    self,
#    #    instance: _AnimatableT
#    #) -> None:
#    #    super().__init__()
#    #    self._instance: _AnimatableT = instance

#    @abstractmethod
#    def update(
#        self,
#        alpha: float
#    ) -> None:
#        pass

#    @abstractmethod
#    def initial_update(self) -> None:
#        pass

#    @abstractmethod
#    def final_update(self) -> None:
#        pass


#class AnimatableSetAnimation(Animation):
#    __slots__ = (
#        "_animatable",
#        "_descriptor",
#        "_elements_0",
#        "_elements_1",
#        "_animations"
#    )

#    def __init__(
#        self: Self,
#        animatable: Animatable,
#        descriptor: LazyDescriptor,
#        animatable_input: Any
#    ) -> None:

#        def iter_elements_from_input(
#            animatable_cls: type[Animatable],
#            animatable_input: Any,
#            is_plural: bool
#        ) -> Iterator:
#            if is_plural:
#                for animatable_input_element in animatable_input:
#                    yield animatable_cls._convert_input(animatable_input_element)
#            else:
#                yield animatable_cls._convert_input(animatable_input)

#        super().__init__()
#        assert (element_type := descriptor._element_type) is not None and issubclass(element_type, Animatable)
#        elements_0 = descriptor._get_elements(animatable)
#        elements_1 = tuple(iter_elements_from_input(
#            animatable_cls=element_type,
#            animatable_input=animatable_input,
#            is_plural=descriptor._is_plural
#        ))
#        self._animatable: Animatable = animatable
#        self._descriptor: LazyDescriptor = descriptor
#        self._elements_0: tuple = elements_0
#        self._elements_1: tuple = elements_1
#        self._animations: tuple[Animation, ...] | None = None
#        #self._positions_0_ = positions_0
#        #self._positions_1_ = positions_1
#        #self._edges_ = edges

#    def update(
#        self: Self,
#        alpha: float
#    ) -> None:
#        super().update(alpha)
#        if (animations := self._animations) is None:
#            dst_elements = self._descriptor._get_elements(self._animatable)
#            animations = tuple(itertools.chain.from_iterable(
#                dst_element._animate_cls._iter_interpolate_animations(
#                    dst=dst_element,
#                    src_0=element_0.copy(),
#                    src_1=element_1.copy()
#                )
#                for dst_element, element_0, element_1 in zip(
#                    dst_elements, self._elements_0, self._elements_1, strict=True
#                )
#            ))
#            self._animations = animations
#        for animation in animations:
#            animation.update(alpha)
#            #interpolate_info.interpolate(self._dst, alpha)

#    def update_boundary(
#        self: Self,
#        boundary: BoundaryT
#    ) -> None:
#        super().update_boundary(boundary)
#        self._descriptor._set_elements(self._animatable, self._elements_1 if boundary else self._elements_0)


class AnimatableInterpolateAnimation[AnimatableT: Animatable](Animation):
    __slots__ = (
        "_dst",
        "_src_0",
        "_src_1",
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
        self._src_0: AnimatableT = src_0
        self._src_1: AnimatableT = src_1
        #self._interpolate_info: AnimatableInterpolateInfo[AnimatableT] | None = None
        #self._positions_0_ = positions_0
        #self._positions_1_ = positions_1
        #self._edges_ = edges

    @abstractmethod
    def interpolate(
        self: Self,
        dst: AnimatableT,
        alpha: float
    ) -> None:
        pass

    @abstractmethod
    def becomes(
        self: Self,
        dst: AnimatableT,
        src: AnimatableT
    ) -> None:
        pass

    #@Lazy.variable()
    #@staticmethod
    #def _src_0_() -> AnimatableT:
    #    return NotImplemented

    #@Lazy.variable()
    #@staticmethod
    #def _src_1_() -> AnimatableT:
    #    return NotImplemented

    #@Lazy.property()
    #@staticmethod
    #def _interpolate_info_(
    #    src_0: AnimatableT,
    #    src_1: AnimatableT
    #) -> AnimatableInterpolateInfo[AnimatableT]:
    #    return type(src_0)._interpolate(src_0, src_1)

    def update(
        self: Self,
        alpha: float
    ) -> None:
        super().update(alpha)
        self.interpolate(self._dst, alpha)
        #if (interpolate_info := self._interpolate_info) is None:
        #    interpolate_info = type(self._dst)._interpolate(self._src_0, self._src_1)
        #    self._interpolate_info = interpolate_info
        #interpolate_info.interpolate(self._dst, alpha)

    def update_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> None:
        super().update_boundary(boundary)
        self.becomes(self._dst, self._src_1 if boundary else self._src_0)
        #self._dst._copy_lazy_content(self._src_1 if boundary else self._src_0)


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
        #dst._copy_lazy_content(new_dst)

    def update_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> None:
        super().update_boundary(boundary)
        self.update(float(boundary))
        #self._dst._copy_lazy_content(self._src)
