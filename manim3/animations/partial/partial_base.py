from typing import Callable

from ...constants.custom_typing import NP_xf8
from ...mobjects.mobject.mobject import Mobject
from ..animation.animation import Animation


class PartialBase(Animation):
    __slots__ = (
        "_mobject",
        "_original_mobject",
        "_alpha_to_segments"
    )

    def __init__(
        self,
        mobject: Mobject,
        alpha_to_segments: Callable[[float], tuple[NP_xf8, list[int]]],
        run_alpha: float = float("inf")
    ) -> None:
        super().__init__(
            run_alpha=run_alpha
        )
        self._mobject: Mobject = mobject
        self._original_mobject: Mobject = mobject.copy()
        self._alpha_to_segments: Callable[[float], tuple[NP_xf8, list[int]]] = alpha_to_segments

    def updater(
        self,
        alpha: float
    ) -> None:
        split_alphas, concatenate_indices = self._alpha_to_segments(alpha)
        for mobject, original_mobject in zip(
            self._mobject.iter_descendants(),
            self._original_mobject.iter_descendants(),
            strict=True
        ):
            equivalent_cls = type(mobject)._equivalent_cls
            mobjects = [equivalent_cls() for _ in range(len(split_alphas) + 1)]
            equivalent_cls._split_into(
                dst_mobject_list=mobjects,
                src_mobject=original_mobject,
                alphas=split_alphas
            )
            equivalent_cls._concatenate_into(
                dst_mobject=mobject,
                src_mobject_list=[mobjects[index] for index in concatenate_indices]
            )
