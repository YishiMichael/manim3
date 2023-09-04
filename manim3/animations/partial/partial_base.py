from typing import Callable

from ...constants.custom_typing import NP_xf8
#from ...mobjects.mobject.operation_handlers.split_bound_handler import SplitBoundHandler
from ...mobjects.mobject.mobject import Mobject
from ..animation.animation import Animation


class PartialBase(Animation):
    __slots__ = (
        "_mobject",
        "_original_state_dict",
        "_alpha_to_split_alphas",
        "_concatenate_indices"
        #"_split_bound_handlers",
        #"_mobject",
        #"_alpha_to_boundary_values",
        #"_backwards"
    )

    def __init__(
        self,
        mobject: Mobject,
        alpha_to_split_alphas: Callable[[float], NP_xf8],
        concatenate_indices: list[int]
        #*,
        #backwards: bool = False
    ) -> None:
        super().__init__(
            run_alpha=1.0
        )
        self._mobject: Mobject = mobject
        self._original_state_dict: dict[Mobject, Mobject] = {
            descendant: descendant._state_copy()
            for descendant in mobject.iter_descendants()
        }
        self._alpha_to_split_alphas: Callable[[float], NP_xf8] = alpha_to_split_alphas
        self._concatenate_indices: list[int] = concatenate_indices
        #self._split_bound_handlers: list[SplitBoundHandler] = [
        #    SplitBoundHandler(descendant, descendant)
        #    for descendant in mobject.iter_descendants()
        #]
        #self._mobject: Mobject = mobject
        #self._alpha_to_boundary_values: Callable[[float], tuple[float, float]] = alpha_to_boundary_values
        #self._backwards: bool = backwards

    def updater(
        self,
        alpha: float
    ) -> None:
        alpha_to_split_alphas = self._alpha_to_split_alphas
        concatenate_indices = self._concatenate_indices
        for mobject, original_state in self._original_state_dict.items():
            equivalent_cls = type(original_state)
            split_alphas = alpha_to_split_alphas(alpha)
            split_mobject_list = [equivalent_cls() for _ in range(len(split_alphas) + 1)]
            equivalent_cls._cls_split(
                dst_mobject_list=split_mobject_list,
                src_mobject=original_state,
                alphas=split_alphas
            )
            equivalent_cls._cls_concatenate(
                dst_mobject=mobject,
                src_mobject_list=[split_mobject_list[index] for index in concatenate_indices]
            )
        #alpha_0, alpha_1 = self._alpha_to_boundary_values(alpha)
        ##if self._backwards:
        ##    alpha_0, alpha_1 = 1.0 - alpha_1, 1.0 - alpha_0
        #for split_bound_handler in self._split_bound_handlers:
        #    split_bound_handler.split(alpha_0, alpha_1)

    async def timeline(self) -> None:
        await self.wait()
