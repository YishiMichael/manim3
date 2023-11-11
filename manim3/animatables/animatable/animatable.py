from __future__ import annotations


from abc import abstractmethod
from typing import (
    #Any,
    #Callable,
    #ClassVar,
    Iterator,
    #Never,
    Self,
    Unpack
)

from ...constants.custom_typing import (
    BoundaryT,
    NP_xf8
)
#from ...lazy.lazy_descriptor import LazyDescriptor
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

    #_actions_cls: type[AnimatableActions] = AnimatableActions
    #_animatable_descriptors: ClassVar[tuple[LazyDescriptor, ...]] = ()
    #_descriptor_converter_dict: ClassVar[dict[str, tuple[LazyDescriptor, Callable[[Any], Animatable]]]] = {}

    #def __init_subclass__(
    #    cls: type[Self]
    #) -> None:
    #    super().__init_subclass__()
    #    #actions_cls: type[AnimatableActions] | None = None
    #    #for base in cls.__mro__:
    #    #    if issubclass(base, AnimatableActions) and not issubclass(base, Animatable):
    #    #        actions_cls = base
    #    #        break
    #    #assert actions_cls is not None

    #    #cls._actions_cls = actions_cls
    #    #cls._animatable_descriptors = tuple(
    #    #    descriptor
    #    #    for descriptor in cls._lazy_descriptors
    #    #    if descriptor in AnimatableMeta._animatable_descriptors
    #    #)
    #    #cls._descriptor_converter_dict = {
    #    #    descriptor._name: (descriptor, converter)
    #    #    for descriptor in cls._lazy_descriptors
    #    #    if (converter := AnimatableMeta._descriptor_converter_dict.get(descriptor)) is not None
    #    #}

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
    ) -> DynamicAnimatable[Self]:
        return DynamicAnimatable(self, **kwargs)

    interpolate = AnimatableActions.interpolate.build_animatable_method_descriptor()
    piecewise = AnimatableActions.piecewise.build_animatable_method_descriptor()
    transform = AnimatableActions.transform.build_animatable_method_descriptor()


class DynamicAnimatable[AnimatableT: Animatable](AnimationsTimeline):
    __slots__ = ("_dst",)

    def __init__(
        self: Self,
        dst: AnimatableT,
        **kwargs: Unpack[AnimateKwargs]
    ) -> None:
        super().__init__(**kwargs)
        self._dst: AnimatableT = dst

    interpolate = AnimatableActions.interpolate.build_dynamic_animatable_method_descriptor()
    piecewise = AnimatableActions.piecewise.build_dynamic_animatable_method_descriptor()
    transform = AnimatableActions.transform.build_dynamic_animatable_method_descriptor()


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

    def update(
        self: Self,
        alpha: float
    ) -> None:
        super().update(alpha)
        self.interpolate(self._dst, alpha)

    def update_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> None:
        super().update_boundary(boundary)
        self.becomes(self._dst, self._src_1 if boundary else self._src_0)


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

    def update_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> None:
        super().update_boundary(boundary)
        self.update(float(boundary))


#class AnimatableMeta:
#    __slots__ = ()

#    _animatable_descriptors: list[LazyDescriptor] = []
#    _descriptor_converter_dict: dict[LazyDescriptor, Callable[[Any], Animatable]] = {}

#    def __new__(
#        cls: type[Self]
#    ) -> Never:
#        raise TypeError

#    @classmethod
#    def register_descriptor[AnimatableT: Animatable, DataT](
#        cls: type[Self]
#    ) -> Callable[[LazyDescriptor[AnimatableT, DataT]], LazyDescriptor[AnimatableT, DataT]]:

#        def result(
#            descriptor: LazyDescriptor[AnimatableT, DataT]
#        ) -> LazyDescriptor[AnimatableT, DataT]:
#            assert not descriptor._is_property
#            assert not descriptor._freeze
#            assert descriptor._deepcopy
#            cls._animatable_descriptors.append(descriptor)
#            return descriptor

#        return result

#    @classmethod
#    def register_converter[AnimatableT: Animatable](
#        cls: type[Self],
#        converter: Callable[[Any], AnimatableT] | None = None
#    ) -> Callable[[LazyDescriptor[AnimatableT, AnimatableT]], LazyDescriptor[AnimatableT, AnimatableT]]:

#        def identity(
#            element: AnimatableT
#        ) -> AnimatableT:
#            return element

#        if converter is None:
#            converter = identity

#        def result(
#            descriptor: LazyDescriptor[AnimatableT, AnimatableT]
#        ) -> LazyDescriptor[AnimatableT, AnimatableT]:
#            assert not descriptor._is_property
#            assert not descriptor._plural
#            assert not descriptor._freeze
#            cls._descriptor_converter_dict[descriptor] = converter
#            return descriptor

#        return result
