from __future__ import annotations


from typing import Self

from manim3 import *


class CreateExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        text = (
            Text("Text")
            .set(color=ORANGE, opacity=0.5)
            .add_strokes(color=BLUE, weight=10.0)
        )
        await self.wait()
        await self.play(Create(text), rate=Rates.smooth(), run_time=2.0)
        await self.wait()
        await self.play(Uncreate(text, backwards=True), rate=Rates.smooth(), run_time=2.0)
        await self.wait()


if __name__ == "__main__":
    with (
        Config(),
        Toplevel.livestream()
    ):
        CreateExample().run()
