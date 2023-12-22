from __future__ import annotations


from typing import Self

from manim3 import *


class WriteAnimationExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        text = Text("Hello").scale(1.5)
        text.move_to(self.camera.frame, DR)
        await self.play(Parallel(*(
            Series(
                Create(stroke := glyph.build_stroke()),
                Parallel(
                    FadeIn(glyph),
                    FadeOut(stroke)
                )
            )
            for glyph in text
            if isinstance(glyph, ShapeMobject)
        ), lag_ratio=0.3), run_time=3.0)
        await self.wait(2.0)


if __name__ == "__main__":
    with (
        Config(),
        Toplevel.livestream()
    ):
        WriteAnimationExample().run()
