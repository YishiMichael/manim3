from typing import Callable

from ..custom_typing import NP_3f8
from ..mobjects.mobject import (
    AboutABC,
    Mobject
)
from ..utils.model_interpolant import ModelInterpolant
from ..utils.rate import RateUtils
from .animation import Animation


class ModelFiniteAnimation(Animation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        model_interpolant: ModelInterpolant,
        about: AboutABC | None = None,
        *,
        arrive: bool = False,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
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
            run_time=run_time,
            relative_rate=RateUtils.adjust(rate_func, run_time_scale=run_time)
        )

    async def timeline(self) -> None:
        await self.wait()


class Shift(ModelFiniteAnimation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        vector: NP_3f8,
        *,
        arrive: bool = False,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ModelInterpolant.from_shift(vector),
            arrive=arrive,
            run_time=run_time,
            rate_func=rate_func
        )


class Scale(ModelFiniteAnimation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        scale: float | NP_3f8,
        about: AboutABC | None = None,
        *,
        arrive: bool = False,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ModelInterpolant.from_scale(scale),
            about=about,
            arrive=arrive,
            run_time=run_time,
            rate_func=rate_func
        )


class Rotate(ModelFiniteAnimation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        rotvec: NP_3f8,
        about: AboutABC | None = None,
        *,
        arrive: bool = False,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ModelInterpolant.from_rotate(rotvec),
            about=about,
            arrive=arrive,
            run_time=run_time,
            rate_func=rate_func
        )


class ModelRunningAnimation(Animation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        model_interpolant: ModelInterpolant,
        about: AboutABC | None = None,
        *,
        run_time: float | None = None,
        speed: float = 1.0
    ) -> None:
        callback = mobject._apply_transform_callback(model_interpolant, about)
        super().__init__(
            updater=callback,
            run_time=run_time,
            relative_rate=RateUtils.adjust(RateUtils.linear, run_alpha_scale=speed)
        )

    async def timeline(self) -> None:
        await self.wait_forever()


class Shifting(ModelRunningAnimation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        vector: NP_3f8,
        *,
        run_time: float | None = None,
        speed: float = 1.0
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ModelInterpolant.from_shift(vector),
            run_time=run_time,
            speed=speed
        )


class Scaling(ModelRunningAnimation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        factor: float | NP_3f8,
        about: AboutABC | None = None,
        *,
        run_time: float | None = None,
        speed: float = 1.0
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ModelInterpolant.from_scale(factor),
            about=about,
            run_time=run_time,
            speed=speed
        )


class Rotating(ModelRunningAnimation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        rotvec: NP_3f8,
        about: AboutABC | None = None,
        *,
        run_time: float | None = None,
        speed: float = 1.0
    ) -> None:
        super().__init__(
            mobject=mobject,
            model_interpolant=ModelInterpolant.from_rotate(rotvec),
            about=about,
            run_time=run_time,
            speed=speed
        )
