from __future__ import annotations


from typing import Self

from manim3 import *


class TextTransformExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        text = (
            Text("Text", concatenate=True)
            .set(color=ORANGE, opacity=0.5)
            .scale(2.0)
            .shift(2.0 * LEFT)
            .add_strokes(color=BLUE, weight=10.0)
        )
        typst = (
            Text("Typst", concatenate=True)
            .set(color=BLUE, opacity=0.5)
            .scale(2.0)
            .add_strokes(color=PINK, weight=10.0)
        )
        code = (
            Code("print(\"Code!\")")
            .scale(2.0)
            .shift(2.0 * RIGHT)
        )
        self.add(text)
        await self.wait()
        await self.play(Transform(text, typst), run_time=2.0, rate=Rates.smooth())
        await self.wait()
        await self.play(FadeTransform(typst, code), run_time=2.0, rate=Rates.smooth())
        await self.wait(3.0)


if __name__ == "__main__":
    with (
        Config(),
        Toplevel.livestream()
    ):
        TextTransformExample().run()
