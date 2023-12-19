from __future__ import annotations


from typing import Self

from manim3 import *


class ShapeTransformExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        square = (
            Square()
            .set(color=WHITE, opacity=1.0)
            .add_strokes(color=YELLOW, width=0.0)
        )
        circle = (
            Circle()
            .set(color=PINK, opacity=0.9)
            .add_strokes(color=YELLOW, weight=10.0)
        )
        triangle = (
            RegularPolygon(3)
            .rotate_about_origin(OUT * (PI / 2.0))
            .set(color=BLUE, opacity=0.9)
            .add_strokes(color=GREEN_A, weight=10.0)
        )

        self.add(square)
        await self.wait()
        await self.play(Transform(square, circle), run_time=2.0, rate=Rates.smooth())
        await self.wait()
        await self.play(Transform(circle, triangle), run_time=2.0, rate=Rates.smooth())
        await self.wait()


if __name__ == "__main__":
    with (
        Config(),
        Toplevel.livestream()
    ):
        ShapeTransformExample().run()
