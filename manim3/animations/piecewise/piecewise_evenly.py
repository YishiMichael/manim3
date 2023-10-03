#from typing import Callable

#import numpy as np

#from ...constants.custom_typing import NP_xf8
#from ...mobjects.mobject import Mobject
#from .piecewise_base import PiecewiseBase


#class PiecewiseEvenly(PiecewiseBase):
#    __slots__ = ()

#    def __init__(
#        self,
#        mobject: Mobject,
#        # Requires that the output `a, b` shall always satisfy:
#        # 0 <= a <= 1, 0 <= b - a <= 1
#        alpha_to_boundaries: Callable[[float], tuple[float, float]],
#        backwards: bool = False,
#        n_segments: int = 1,
#        run_alpha: float = float("inf")
#    ) -> None:
#        assert n_segments > 0

#        def alpha_to_segments(
#            alpha: float
#        ) -> tuple[NP_xf8, list[int]]:
#            boundaries = np.array(alpha_to_boundaries(alpha))
#            split_alphas = np.linspace(boundaries, boundaries + 1.0, n_segments, endpoint=False).flatten()
#            if split_alphas[-1] > 1.0:
#                split_alphas = np.roll(split_alphas, 1)
#                split_alphas[0] -= 1.0
#                concatenate_indices = list(range(0, 2 * n_segments + 1, 2))
#            else:
#                concatenate_indices = list(range(1, 2 * n_segments + 1, 2))
#            if backwards:
#                split_alphas = 1.0 - split_alphas[::-1]
#            return split_alphas, concatenate_indices

#        super().__init__(
#            mobject=mobject,
#            alpha_to_segments=alpha_to_segments,
#            run_alpha=run_alpha
#        )