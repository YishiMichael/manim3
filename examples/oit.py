from __future__ import annotations


from typing import Self

import numpy as np
from manim3 import *


class OITExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        self.add(*(
            (Circle()
                .set(color=color, opacity=opacity, weight=weight)
                .shift(0.5 * RIGHT)
                .rotate_about_origin(angle * OUT)
            )
            for color, opacity, weight, angle in zip(
                (RED, GREEN, BLUE),
                (0.3, 0.5, 0.6),
                (1.0, 2.0, 1.0),
                np.linspace(0.0, TAU, 3, endpoint=False),
                strict=True
            )
        ))
        await self.wait(5.0)


if __name__ == "__main__":
    with (
        Config(),
        Toplevel.livestream()
    ):
        OITExample().run()
