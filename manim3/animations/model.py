from typing import Callable

from scipy.spatial.transform import Rotation

from ..animations.animation import Animation
from ..custom_typing import (
    Mat4T,
    Vec3T
)
from ..mobjects.mobject import (
    AboutABC,
    Mobject
)
from ..utils.rate import RateUtils


class ModelAnimationABC(Animation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        alpha_to_matrix: Callable[[float], Mat4T],
        *,
        run_time: float | None = None,
        relative_rate: Callable[[float], float] = RateUtils.linear
    ) -> None:
        initial_model_matrix = mobject._model_matrix_.value

        def updater(
            alpha: float
        ) -> None:
            mobject._model_matrix_ = alpha_to_matrix(alpha) @ initial_model_matrix

        super().__init__(
            updater=updater,
            run_time=run_time,
            relative_rate=relative_rate
        )


class Shift(ModelAnimationABC):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        vector: Vec3T,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            mobject=mobject,
            alpha_to_matrix=mobject._shift_callback(vector),
            run_time=run_time,
            relative_rate=RateUtils.adjust(rate_func, run_time_scale=run_time)
        )

    async def timeline(self) -> None:
        await self.wait()


class Scale(ModelAnimationABC):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        scale: float | Vec3T,
        about: AboutABC | None = None,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            mobject=mobject,
            alpha_to_matrix=mobject._scale_callback(scale, about),
            run_time=run_time,
            relative_rate=RateUtils.adjust(rate_func, run_time_scale=run_time)
        )

    async def timeline(self) -> None:
        await self.wait()


class Rotate(ModelAnimationABC):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        rotation: Rotation,
        about: AboutABC | None = None,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            mobject=mobject,
            alpha_to_matrix=mobject._rotate_callback(rotation, about),
            run_time=run_time,
            relative_rate=RateUtils.adjust(rate_func, run_time_scale=run_time)
        )

    async def timeline(self) -> None:
        await self.wait()


class Shifting(ModelAnimationABC):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        vector: Vec3T,
        *,
        speed: float = 1.0,
        run_time: float | None = None
    ) -> None:
        super().__init__(
            mobject=mobject,
            alpha_to_matrix=mobject._shift_callback(vector),
            run_time=run_time,
            relative_rate=RateUtils.adjust(RateUtils.linear, run_alpha_scale=speed)
        )


class Scaling(ModelAnimationABC):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        factor: float | Vec3T,
        about: AboutABC | None = None,
        *,
        speed: float = 1.0,
        run_time: float | None = None
    ) -> None:
        super().__init__(
            mobject=mobject,
            alpha_to_matrix=mobject._scale_callback(factor, about),
            run_time=run_time,
            relative_rate=RateUtils.adjust(RateUtils.linear, run_alpha_scale=speed)
        )


class Rotating(ModelAnimationABC):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        rotation: Rotation,
        about: AboutABC | None = None,
        *,
        speed: float = 1.0,
        run_time: float | None = None
    ) -> None:
        super().__init__(
            mobject=mobject,
            alpha_to_matrix=mobject._rotate_callback(rotation, about),
            run_time=run_time,
            relative_rate=RateUtils.adjust(RateUtils.linear, run_alpha_scale=speed)
        )
