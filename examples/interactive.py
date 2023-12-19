from __future__ import annotations


from typing import Self

import numpy as np
from manim3 import *


class InteractiveExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        squares = ShapeMobject().add(*(
            (
                Circle()
                .scale(0.5)
                .shift(x * RIGHT)
                .set(color=color)
            )
            for x, color in zip(
                np.linspace(-4.0, 4.0, 5),
                (RED, YELLOW, GREEN, BLUE, PURPLE),
                strict=True
            )
        ))
        text = Text("Press space to animate.").shift(1.5 * DOWN)
        self.add(squares, text)
        timelines = tuple(
            square.animate().shift(UP)
            for square in squares
        )
        for timeline in timelines:
            self.prepare(
                timeline,
                rate=Rates.smooth(),
                launch_condition=KeyPress(KEY.SPACE).captured
            )
        await self.wait_until(lambda: all(timeline.terminated() for timeline in timelines))
        await self.wait()


if __name__ == "__main__":
    with (
        Config(),
        Toplevel.livestream()
    ):
        InteractiveExample().run()
