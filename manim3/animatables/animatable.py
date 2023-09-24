from abc import abstractmethod
from typing import (
    #TYPE_CHECKING,
    Generic,
    TypeVar
)

from ..animations.animation.animation import Animation
from ..lazy.lazy_object import LazyObject

#if TYPE_CHECKING:
#    from typing_extensions import Self


_T = TypeVar("_T", bound="Animatable")


class Animatable(LazyObject):
    __slots__ = (
        "_is_constructing_animation",
        "_updaters"
    )

    def __init__(self) -> None:
        super().__init__()
        self._is_constructing_animation: bool = False
        self._updaters: "list[Updater]" = []

    def _handle_updater(
        self,
        updater: "Updater"
    ):
        if self._is_constructing_animation:
            self._updaters.append(updater)
        else:
            updater.update(1.0)
        return self

    @property
    def animate(self):
        assert not self._is_constructing_animation
        self._is_constructing_animation = True
        return self

    def build(
        self,
        run_alpha: float = 1.0,
        rewind: bool = False
    ) -> "UpdaterAnimation":
        assert self._is_constructing_animation
        animation = UpdaterAnimation(
            #instance=self,
            updaters=self._updaters.copy(),
            run_alpha=run_alpha,
            rewind=rewind
        )
        self._is_constructing_animation = False
        self._updaters.clear()
        return animation


class Updater(LazyObject):
    __slots__ = ("_instance",)

    def __init__(
        self,
        instance: Animatable
    ) -> None:
        super().__init__()
        self._instance: Animatable = instance

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
        "_rewind"
    )

    def __init__(
        self,
        #instance: _T,
        updaters: list[Updater],
        run_alpha: float,
        rewind: bool
    ) -> None:
        super().__init__(run_alpha=run_alpha)
        #self._instance: _T = instance
        self._updaters: list[Updater] = updaters
        self._rewind: bool = rewind

    def update(
        self,
        alpha: float
    ) -> None:
        adjusted_alpha = self._run_alpha - alpha if self._rewind else alpha
        #instance = self._instance
        for updater in self._updaters:
            updater.update(adjusted_alpha)

    async def timeline(self) -> None:
        await self.wait(self._run_alpha)
