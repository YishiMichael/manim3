from abc import abstractmethod
from typing import Any, Generic
from typing import (
#    #TYPE_CHECKING,
#    Callable,
#    Concatenate,
#    #Generic,
#    Iterator,
#    ParamSpec,
    TypeVar
)

from ..animations.animation.animation import Animation
from ..animations.animation.rates.linear import Linear
from ..animations.animation.rates.rate import Rate
from ..lazy.lazy_object import LazyObject

#if TYPE_CHECKING:
#    from typing_extensions import Self


_AnimatableT = TypeVar("_AnimatableT", bound="Animatable")
#_P = ParamSpec("_P")


class Animatable(LazyObject):
    __slots__ = (
        "_saved_state",
        "_updaters"
    )

    def __init__(self) -> None:
        super().__init__()
        self._saved_state: Animatable | None = None
        self._updaters: "list[Updater]" = []

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

    @classmethod
    def _convert_input(
        cls: type[_AnimatableT],
        animatable_input: Any
    ) -> _AnimatableT:
        return animatable_input

    def _get_interpolate_updater(
        self: _AnimatableT,
        animatable_0: _AnimatableT,
        animatable_1: _AnimatableT
    ) -> "Updater[_AnimatableT]":
        raise NotImplementedError

    def _stack_updater(
        self,
        updater: "Updater"
    ) -> None:
        updater.update(1.0)
        if self._saved_state is not None:
            self._updaters.append(updater)

    def interpolate(
        self,
        animatable_0: _AnimatableT,
        animatable_1: _AnimatableT
    ):
        self._stack_updater(self._get_interpolate_updater(
            animatable_0, animatable_1
        ))
        return self

    def set(
        self,
        **kwargs: Any
    ):
        for attribute_name, animatable_input in kwargs.items():
            descriptor = self._lazy_descriptors[f"_{attribute_name}_"]
            animatable_cls = descriptor._element_type
            assert issubclass(animatable_cls, Animatable)
            target_animatable = animatable_cls._convert_input(animatable_input)
            source_animatable = descriptor.__get__(self)
            descriptor.__set__(self, target_animatable)
            if self._saved_state is not None:
                self._updaters.append(source_animatable._get_interpolate_updater(
                    source_animatable._copy(), target_animatable._copy()
                ))
        return self

    @property
    def animate(self):
        assert self._saved_state is None
        self._saved_state = self._copy()
        return self

    def build(
        self,
        rate: Rate = Linear(),
        #run_alpha: float = 1.0,
        infinite: bool = False,
        #rewind: bool = False
    ) -> "UpdaterAnimation":
        #assert not infinite or not rewind
        assert (saved_state := self._saved_state) is not None
        self._copy_lazy_content(self, saved_state)
        animation = UpdaterAnimation(
            #instance=self,
            updaters=self._updaters.copy(),
            rate=rate,
            run_alpha=float("inf") if infinite else 1.0
            #infinite=infinite,
            #run_alpha=float("inf") if infinite else 1.0,
            #rewind=rewind
        )
        self._saved_state = None
        self._updaters.clear()
        return animation


class Updater(Generic[_AnimatableT], LazyObject):
    __slots__ = ("_instance",)

    def __init__(
        self,
        instance: _AnimatableT
    ) -> None:
        super().__init__()
        self._instance: _AnimatableT = instance

    @abstractmethod
    def update(
        self,
        alpha: float
    ) -> None:
        pass


class UpdaterAnimation(Animation):
    __slots__ = (
        #"_instance",
        "_updaters",
        "_rate"
    )

    def __init__(
        self,
        #instance: _T,
        updaters: list[Updater],
        rate: Rate,
        run_alpha: float
    ) -> None:
        super().__init__(run_alpha=run_alpha)
        #self._instance: _T = instance
        self._updaters: list[Updater] = updaters
        self._rate: Rate = rate

    def update(
        self,
        alpha: float
    ) -> None:
        sub_alpha = self._rate.at(alpha)
        #instance = self._instance
        for updater in self._updaters:
            updater.update(sub_alpha)

    async def timeline(self) -> None:
        await self.wait(self._run_alpha)
