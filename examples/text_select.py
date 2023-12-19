from __future__ import annotations


from typing import Self

from manim3 import *


class TextSelectExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        text = Text("Typst Text").shift(UP)
        text.select("Typst").set(color=TEAL)

        equation = Math("a^2 + b^2 = c^2").shift(DOWN)
        equation.set_local_colors({
            "a": GREEN,
            "b": RED,
            "c": BLUE
        })

        self.add(text, equation)
        await self.wait(5.0)


if __name__ == "__main__":
    with (
        Config(),
        Toplevel.livestream()
    ):
        TextSelectExample().run()
