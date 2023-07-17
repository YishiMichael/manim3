from typing import (
    Callable,
    TypeVar
)

from ..mobjects.mobject import Mobject
from ..mobjects.mobject_style_meta import MobjectStyleMeta
from .animation import Animation


_MobjectT = TypeVar("_MobjectT", bound=Mobject)


class TransformBase(Animation):
    __slots__ = (
        "_start_mobject",
        "_stop_mobject",
        "_intermediate_mobject"
    )

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject,
        intermediate_mobject: Mobject
    ) -> None:
        callbacks = tuple(
            MobjectStyleMeta._interpolate(start_descendant, stop_descendant)(intermediate_descendant)
            for start_descendant, stop_descendant, intermediate_descendant in zip(
                start_mobject.iter_descendants(),
                stop_mobject.iter_descendants(),
                intermediate_mobject.iter_descendants(),
                strict=True
            )
        )

        def updater(
            alpha: float
        ) -> None:
            for callback in callbacks:
                callback(alpha)

        super().__init__(
            updater=updater,
            run_alpha=1.0
        )
        self._start_mobject: Mobject = start_mobject
        self._stop_mobject: Mobject = stop_mobject
        self._intermediate_mobject: Mobject = intermediate_mobject

    async def timeline(self) -> None:
        await self.wait()


class TransformTo(TransformBase):
    __slots__ = ()

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject
    ) -> None:
        super().__init__(
            start_mobject=start_mobject,
            stop_mobject=stop_mobject,
            intermediate_mobject=start_mobject
        )


class TransformFrom(TransformBase):
    __slots__ = ()

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject
    ) -> None:
        super().__init__(
            start_mobject=start_mobject,
            stop_mobject=stop_mobject,
            intermediate_mobject=stop_mobject
        )


class TransformToCopy(TransformTo):
    __slots__ = ()

    def __init__(
        self,
        mobject: _MobjectT,
        func: Callable[[_MobjectT], _MobjectT]
    ) -> None:
        super().__init__(
            start_mobject=mobject,
            stop_mobject=func(mobject.copy())
        )


class TransformFromCopy(TransformFrom):
    __slots__ = ()

    def __init__(
        self,
        mobject: _MobjectT,
        func: Callable[[_MobjectT], _MobjectT]
    ) -> None:
        super().__init__(
            start_mobject=func(mobject.copy()),
            stop_mobject=mobject
        )


class Transform(TransformBase):
    __slots__ = ()

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject
    ) -> None:
        super().__init__(
            start_mobject=start_mobject,
            stop_mobject=stop_mobject,
            intermediate_mobject=start_mobject.copy()
        )

    async def timeline(self) -> None:
        self.scene.discard(self._start_mobject)
        self.scene.add(self._intermediate_mobject)
        await super().timeline()
        self.scene.discard(self._intermediate_mobject)
        self.scene.add(self._stop_mobject)
