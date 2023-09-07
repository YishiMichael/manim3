from typing import Callable

from ...constants.custom_typing import NP_xf8
#from ...mobjects.mobject.operation_handlers.split_bound_handler import SplitBoundHandler
from ...mobjects.mobject.mobject import Mobject
from ..animation.animation import Animation


class PartialBase(Animation):
    __slots__ = (
        "_mobject",
        "_original_mobject",
        "_alpha_to_segments"
        #"_alpha_to_concatenate_indices"
        #"_split_bound_handlers",
        #"_mobject",
        #"_alpha_to_boundary_values",
        #"_backwards"
    )

    def __init__(
        self,
        mobject: Mobject,
        alpha_to_segments: Callable[[float], tuple[NP_xf8, list[int]]],
        run_alpha: float = float("inf")
        #*,
        #backwards: bool = False
    ) -> None:
        super().__init__(
            run_alpha=run_alpha
        )
        #print(mobject._graph_._edges_)
        self._mobject: Mobject = mobject
        self._original_mobject: Mobject = mobject.copy()
        self._alpha_to_segments: Callable[[float], tuple[NP_xf8, list[int]]] = alpha_to_segments
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
        #print(self._mobject)
        #print(self._original_mobject)
        split_alphas, concatenate_indices = self._alpha_to_segments(alpha)
        for mobject, original_mobject in zip(
            self._mobject.iter_descendants(),
            self._original_mobject.iter_descendants(),
            strict=True
        ):
            equivalent_cls = type(mobject)._equivalent_cls
            mobjects = [equivalent_cls() for _ in range(len(split_alphas) + 1)]
            #print(mobject, len(mobject._graph_._edges_))
            equivalent_cls._split_into(
                dst_mobject_list=mobjects,
                src_mobject=original_mobject,
                alphas=split_alphas
            )
            #print(self._mobject._lazy_slots)
            #print(self._original_mobject._lazy_slots)
            #print(self._mobject._graph_)
            #print(self._original_mobject._graph_)
            equivalent_cls._concatenate_into(
                dst_mobject=mobject,
                src_mobject_list=[mobjects[index] for index in concatenate_indices]
            )
            #print(self._mobject._graph_)
            #print(self._original_mobject._graph_)
            #print()
        #alpha_0, alpha_1 = self._alpha_to_boundary_values(alpha)
        ##if self._backwards:
        ##    alpha_0, alpha_1 = 1.0 - alpha_1, 1.0 - alpha_0
        #for split_bound_handler in self._split_bound_handlers:
        #    split_bound_handler.split(alpha_0, alpha_1)
