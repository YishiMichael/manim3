from __future__ import annotations


from typing import Self

from manim3 import *


class ThreeDExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        dodec = (
            Dodecahedron()
            .scale(2.0)
            .set(
                color="#00FFAA",
                opacity=0.25
            )
            .bind_lighting(Lighting(
                AmbientLight().set(color=GREY_D),
                PointLight().shift(5.0 * RIGHT)
            ))
        )
        self.add(dodec)
        self.prepare(self.camera.animate(infinite=True).rotate(0.5 * DOWN))

        text = Text("Dodecahedron")
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
        ThreeDExample().run()
