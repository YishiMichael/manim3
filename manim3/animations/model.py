from ..constants.custom_typing import NP_3f8
from ..mobjects.mobject import (
    AboutABC,
    Mobject
)
from ..utils.model_interpolant import ModelInterpolant
from .animation import Animation


class ModelFiniteAnimationBase(Animation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        model_interpolant: ModelInterpolant,
        about: AboutABC | None = None,
        *,
        arrive: bool = False
    ) -> None:
        callback = mobject._apply_transform_callback(model_interpolant, about)

        def updater(
            alpha: float
        ) -> None:
            if arrive:
                alpha -= 1.0
            callback(alpha)

        super().__init__(
            updater=updater,
            run_alpha=1.0
        )

    async def timeline(self) -> None:
        await self.wait()


class Shift(ModelFiniteAnimationBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        vector: NP_3f8,
        *,
        arrive: bool = False
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ModelInterpolant.from_shift(vector),
            arrive=arrive
        )


class Scale(ModelFiniteAnimationBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        scale: float | NP_3f8,
        about: AboutABC | None = None,
        *,
        arrive: bool = False
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ModelInterpolant.from_scale(scale),
            about=about,
            arrive=arrive
        )


class Rotate(ModelFiniteAnimationBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        rotvec: NP_3f8,
        about: AboutABC | None = None,
        *,
        arrive: bool = False
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ModelInterpolant.from_rotate(rotvec),
            about=about,
            arrive=arrive
        )


class ModelInfiniteAnimationBase(Animation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        model_interpolant: ModelInterpolant,
        about: AboutABC | None = None
    ) -> None:
        callback = mobject._apply_transform_callback(model_interpolant, about)
        super().__init__(
            updater=callback
        )

    async def timeline(self) -> None:
        await self.wait_forever()


class Shifting(ModelInfiniteAnimationBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        vector: NP_3f8
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ModelInterpolant.from_shift(vector)
        )


class Scaling(ModelInfiniteAnimationBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        factor: float | NP_3f8,
        about: AboutABC | None = None
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ModelInterpolant.from_scale(factor),
            about=about
        )


class Rotating(ModelInfiniteAnimationBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        rotvec: NP_3f8,
        about: AboutABC | None = None
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ModelInterpolant.from_rotate(rotvec),
            about=about
        )
