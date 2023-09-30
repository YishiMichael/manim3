#from typing import Callable

#from ...constants.custom_typing import (
#    NP_xf8,
#    NP_xi4
#)
#from ...mobjects.mobject import Mobject
#from ..animation.animation import Animation


#class PiecewiseBase(Animation):
#    __slots__ = (
#        "_mobject",
#        #"_original_mobject",
#        "_piecewise_func",
#        "_infinite"
#    )

#    def __init__(
#        self,
#        mobject: Mobject,
#        piecewise_func: Callable[[float], tuple[NP_xf8, NP_xi4]],
#        #backwards: bool = False,
#        infinite: bool = False
#    ) -> None:
#        super().__init__(
#            run_alpha=float("inf") if infinite else 1.0
#        )
#        self._mobject: Mobject = mobject
#        #self._original_mobject: Mobject = mobject.copy()
#        self._piecewise_func: Callable[[float], tuple[NP_xf8, NP_xi4]] = piecewise_func
#        self._infinite: bool = infinite

#    async def timeline(self) -> None:
#        await self.play(self._mobject.animate.piecewise(self._mobject.copy(), self._piecewise_func).build(infinite=self._infinite))

#    #def update(
#    #    self,
#    #    alpha: float
#    #) -> None:
#    #    split_alphas, concatenate_indices = self._alpha_to_segments(alpha)
#    #    for mobject, original_mobject in zip(
#    #        self._mobject.iter_descendants(),
#    #        self._original_mobject.iter_descendants(),
#    #        strict=True
#    #    ):
#    #        equivalent_cls = type(mobject)._equivalent_cls
#    #        mobjects = [equivalent_cls() for _ in range(len(split_alphas) + 1)]
#    #        equivalent_cls._split_into(
#    #            dst_mobject_list=mobjects,
#    #            src_mobject=original_mobject,
#    #            alphas=split_alphas
#    #        )
#    #        equivalent_cls._concatenate_into(
#    #            dst_mobject=mobject,
#    #            src_mobject_list=[mobjects[index] for index in concatenate_indices]
#    #        )
