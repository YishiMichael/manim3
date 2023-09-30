#from abc import abstractmethod
from typing import (
    #Any,
    Callable,
    ClassVar,
    #ClassVar,
    #Iterator,
    TypeVar
)

from ...animations.animation.rate import Rate
from ...animations.animation.rates import Rates
from ...animations.animation.animation import Animation
#from ..animations.animation.rates.linear import Linear
#from ..animations.animation.rates.rate import Rate
from ...constants.custom_typing import (
    BoundaryT,
    NP_xf8,
    NP_xi4
)
from ...lazy.lazy_descriptor import LazyDescriptor
from ...lazy.lazy_object import LazyObject
from .piecewiser import Piecewiser
from .piecewisers import Piecewisers


_AnimatableT = TypeVar("_AnimatableT", bound="Animatable")
#_P = ParamSpec("_P")


class Animatable(LazyObject):
    __slots__ = (
        "_in_animating_mode",
        "_updater"
    )

    _special_slot_copiers: ClassVar[dict[str, Callable]] = {
        "_in_animating_mode": lambda o: False,
        "_updater": lambda o: Updater()
    }

    _animatable_descriptors: ClassVar[dict[str, LazyDescriptor]] = {}

    #_unanimatable_variable_names: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._animatable_descriptors = {
            name: descriptor
            for name, descriptor in cls._lazy_descriptors.items()
            if descriptor._is_variable
            and descriptor._element_type is not None
            and issubclass(descriptor._element_type, Animatable)
            and descriptor._element_type is not cls
        }

    def __init__(self) -> None:
        super().__init__()
        self._in_animating_mode: bool = False
        self._updater: Updater = Updater()

    #@classmethod
    #def _stack_updaters(
    #    cls,
    #    method: "Callable[Concatenate[_T, _P], Iterator[Updater]]"
    #) -> Callable[Concatenate[_T, _P], _T]:

    #    def result(
    #        self: _T,
    #        *args: _P.args,
    #        **kwargs: _P.kwargs
    #    ) -> _T:
    #        for updater in method(self, *args, **kwargs):
    #            self._updaters.append(updater)
    #            updater.update(1.0)
    #        return self

    #    return result

    #@classmethod
    #@property
    #def _animatable_descriptors(cls) -> Iterator[LazyDescriptor]:
    #    for descriptor in cls._lazy_descriptors.values():
    #        if descriptor._is_variable and descriptor._element_type is not cls:
    #            yield descriptor

    @classmethod
    def _convert_input(
        cls: type[_AnimatableT],
        animatable_input: _AnimatableT
    ) -> _AnimatableT:
        return animatable_input

    def _stack_updater(
        self,
        updater: "Updater"
    ) -> None:
        updater.update_boundary(1)
        #if self._saved_state is not None:
        self._updater.add(updater)

    def _get_interpolate_updater(
        self: _AnimatableT,
        src_0: _AnimatableT,
        src_1: _AnimatableT
    ) -> "Updater":
        updater = Updater()
        for descriptor in type(self)._animatable_descriptors.values():
            #assert issubclass(descriptor._element_type, Animatable)
            dst_elements = descriptor._get_elements(self)
            src_0_elements = descriptor._get_elements(src_0)
            src_1_elements = descriptor._get_elements(src_1)
            #if not issubclass(element_type, Animatable) and src_0_elements != src_1_elements:
            #    raise ValueError(f"Uninterpolable variables of `{descriptor._name}` don't match")
            updater.add(*(
                dst_element._get_interpolate_updater(src_0_element, src_1_element)
                for dst_element, src_0_element, src_1_element in zip(
                    dst_elements, src_0_elements, src_1_elements, strict=True
                )
            ))
        return updater

    def _get_piecewise_updater(
        self: _AnimatableT,
        src: _AnimatableT,
        piecewiser: Piecewiser
        #split_alphas: NP_xf8,
        #concatenate_indices: NP_xi4
    ) -> "Updater":
        updater = Updater()
        for descriptor in type(self)._animatable_descriptors.values():
            #assert issubclass(descriptor._element_type, Animatable)
            dst_elements = descriptor._get_elements(self)
            src_elements = descriptor._get_elements(src)
            updater.add(*(
                dst_element._get_piecewise_updater(src_element, piecewiser)
                for dst_element, src_element in zip(
                    dst_elements, src_elements, strict=True
                )
            ))
        return updater
        #pieces = tuple(cls() for _ in range(len(split_alphas) + 1))
        #cls._split(pieces, src, split_alphas)
        #cls._concatenate(dst, tuple(pieces[index] for index in concatenate_indices))

    def _get_set_updater(
        self,
        **kwargs
    ) -> "Updater":
        updater = Updater()
        for attribute_name, animatable_input in kwargs.items():
            if (descriptor := type(self)._animatable_descriptors.get(f"_{attribute_name}_")) is None:
                continue
            assert (element_type := descriptor._element_type) is not None and issubclass(element_type, Animatable)
            source_elements = descriptor._get_elements(self)
            if descriptor._is_multiple:
                target_elements = tuple(
                    element_type._convert_input(animatable_input_element)
                    for animatable_input_element in animatable_input
                )
            else:
                target_elements = (element_type._convert_input(animatable_input),)
            #descriptor.__set__(self, target_animatable)
            #if self._saved_state is not None:
            for source_element, target_element in zip(source_elements, target_elements, strict=True):
                updater.add(source_element._get_interpolate_updater(
                    source_element.copy(), target_element.copy()
                ))
        return updater

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
    #    src_tuple: tuple[_AnimatableT, ...]
    #) -> None:
    #    for descriptor in cls._animatable_descriptors.values():
    #        assert issubclass(element_type := descriptor._element_type, Animatable)
    #        for dst_element, src_element_tuple in zip(
    #            descriptor._get_elements(dst),
    #            tuple(descriptor._get_elements(src) for src in src_tuple),
    #            strict=True
    #        ):
    #            element_type._concatenate(dst_element, src_element_tuple)

    #def _get_interpolate_updater(
    #    self: _AnimatableT,
    #    animatable_0: _AnimatableT,
    #    animatable_1: _AnimatableT
    #) -> "Updater":
    #    raise NotImplementedError

    #def interpolate(
    #    self,
    #    animatable_0: _AnimatableT,
    #    animatable_1: _AnimatableT
    #):
    #    self._stack_updater(type(self)._interpolate(
    #        self, animatable_0, animatable_1
    #    ))
    #    return self

    @property
    def animate(self):
        assert not self._in_animating_mode, "Already in animating mode"
        self._in_animating_mode = True
        self._updater = Updater()
        return self

    def build(
        self,
        rate: Rate = Rates.linear(),
        #run_alpha: float = 1.0,
        infinite: bool = False
        #rewind: bool = False
    ) -> "UpdaterAnimation":
        #assert not infinite or not rewind
        #assert (saved_state := self._saved_state) is not None
        #self._copy_lazy_content(self, saved_state)
        assert self._in_animating_mode, "Not in animating mode"
        animation = self._updater.build_animation(
            rate=rate,
            infinite=infinite
        )
        #animation = UpdaterAnimation(
        #    #instance=self,
        #    updater=self._updater,
        #    rate=rate,
        #    run_alpha=float("inf") if infinite else 1.0
        #    #infinite=infinite,
        #    #run_alpha=float("inf") if infinite else 1.0,
        #    #rewind=rewind
        #)
        self._in_animating_mode = False
        self._updater = Updater()
        return animation

    def interpolate(
        self: _AnimatableT,
        animatable_0: _AnimatableT,
        animatable_1: _AnimatableT
    ):
        self._stack_updater(self._get_interpolate_updater(animatable_0, animatable_1))
        return self

    def piecewise(
        self: _AnimatableT,
        animatable: _AnimatableT,
        piecewiser: Piecewiser
    ):
        self._stack_updater(self._get_piecewise_updater(animatable, piecewiser))
        return self

    def set(
        self,
        **kwargs
    ):
        self._stack_updater(self._get_set_updater(**kwargs))
        return self

    def transform(
        self: _AnimatableT,
        animatable: _AnimatableT
    ):
        self.interpolate(self.copy(), animatable)
        return self

    def static_interpolate(
        self: _AnimatableT,
        animatable_0: _AnimatableT,
        animatable_1: _AnimatableT,
        alpha: float
    ):
        self._get_interpolate_updater(animatable_0, animatable_1).update(alpha)
        return self

    def static_piecewise(
        self: _AnimatableT,
        animatable: _AnimatableT,
        split_alphas: NP_xf8,
        concatenate_indices: NP_xi4
    ):
        self._get_piecewise_updater(animatable, Piecewisers.static(split_alphas, concatenate_indices)).update(1.0)
        return self


class LeafUpdaterAnimation(Animation):
    __slots__ = (
        #"_instance",
        "_updater",
        "_rate"
    )

    def __init__(
        self,
        #instance: _T,
        #updaters: "tuple[Updater, ...]",
        updater: "Updater",
        rate: Rate,
        run_alpha: float
    ) -> None:
        super().__init__(run_alpha=run_alpha)
        #self._instance: _T = instance
        self._updater: Updater = updater
        self._rate: Rate = rate

    def _animate_instant(
        self,
        alpha: float
    ) -> None:
        self._updater.restore()
        self._updater.update(self._rate.at(alpha))

    async def timeline(self) -> None:
        await self.wait(self._run_alpha)


class UpdaterAnimation(Animation):
    __slots__ = (
        #"_instance",
        "_updater",
        "_rate"
    )

    def __init__(
        self,
        #instance: _T,
        #updaters: "tuple[Updater, ...]",
        updater: "Updater",
        rate: Rate,
        run_alpha: float
    ) -> None:
        super().__init__(run_alpha=run_alpha)
        #self._instance: _T = instance
        self._updater: Updater = updater
        self._rate: Rate = rate

    #def update(
    #    self,
    #    alpha: float
    #) -> None:
    #    self._updater.update(self._rate.at(alpha))
    #    #sub_alpha = self._rate.at(alpha)
    #    #instance = self._instance
    #    #for updater in self._updaters:
    #    #    updater.update(sub_alpha)

    async def timeline(self) -> None:
        #for updater in reversed(self._updaters):
        #    updater.initial_update()
        self._updater.update_boundary(self._rate.at_boundary(0))
        await self.play(LeafUpdaterAnimation(self._updater, self._rate, self._run_alpha))
        #await self.wait(self._run_alpha)
        self._updater.update_boundary(self._rate.at_boundary(1))
        #for updater in self._updaters:
        #    updater.final_update()


#class Updater(LazyObject):
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


class Updater(LazyObject):
    __slots__ = ("_branch_updaters",)

    def __init__(self) -> None:
        super().__init__()
        self._branch_updaters: "list[Updater]" = []

    def add(
        self,
        *updaters: "Updater"
    ):
        self._branch_updaters.extend(updaters)
        return self

    def build_animation(
        self,
        rate: Rate = Rates.linear(),
        infinite: bool = False
    ) -> UpdaterAnimation:
        return UpdaterAnimation(
            updater=self,
            rate=rate,
            run_alpha=float("inf") if infinite else 1.0
        )

    def update(
        self,
        alpha: float
    ) -> None:
        for updater in self._branch_updaters:
            updater.update(alpha)

    def update_boundary(
        self,
        boundary: BoundaryT
    ) -> None:
        for updater in self._branch_updaters:
            updater.update_boundary(boundary)

    def restore(self) -> None:
        for updater in reversed(self._branch_updaters):
            updater.restore()
