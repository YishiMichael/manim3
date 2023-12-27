from __future__ import annotations


from typing import Self

from manim3 import *


class LaggedAnimationExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        text = Text("Text").scale(2.0)
        await self.play(Parallel(*(
            Parallel(
                FadeIn(char),
                char.animate(rewind=True).shift(DOWN)
            )
            for char in text
        ), lag_time=0.5), rate=Rates.smooth())
        await self.wait(3.0)


if __name__ == "__main__":
    with (
        Config(),
        Toplevel.livestream()
    ):
        LaggedAnimationExample().run()
