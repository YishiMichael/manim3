import re
from typing import Self

import numpy as np
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

        self.add(square)
        await self.wait()
        await self.play(Transform(square, circle), run_time=2.0, rate=Rates.smooth())
        await self.wait()


class TextTransformExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        text = (
            Text("Text", concatenate=True)
            .set(color=ORANGE, opacity=0.5)
            .shift(2.0 * LEFT)
            .add_strokes(color=BLUE, weight=10.0)
        )
        tex = (
            Tex("Tex", concatenate=True)
            .set(color=BLUE, opacity=0.5)
            .add_strokes(color=PINK, weight=10.0)
        )
        code = Code("print(\"Code!\")").shift(2.0 * RIGHT)
        self.add(text)
        await self.wait()
        await self.play(Transform(text, tex), run_time=2.0, rate=Rates.smooth())
        await self.wait()
        await self.play(FadeTransform(tex, code), run_time=2.0, rate=Rates.smooth())
        await self.wait(3.0)


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


class WriteExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        tex = Tex("Hello").scale(2.0)
        await self.play(Parallel(*(
            Series(
                Create(stroke := glyph.build_stroke()),
                Parallel(
                    FadeIn(glyph),
                    FadeOut(stroke)
                )
            )
            for glyph in tex
            if isinstance(glyph, ShapeMobject)
        ), lag_ratio=0.3), run_time=3.0)
        await self.wait(2.0)


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

        text = Text("Dodecahedron").scale(0.5)
        await self.play(Parallel(*(
            Parallel(
                FadeIn(char),
                char.animate(rewind=True).shift(DOWN)
            )
            for char in text
        ), lag_time=0.5), rate=Rates.smooth())
        await self.wait(3.0)


class OITExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        self.add(*(
            (Circle()
                .set(color=color, opacity=opacity)
                .shift(0.5 * RIGHT)
                .rotate_about_origin(angle * OUT)
            )
            for color, opacity, angle in zip(
                (RED, GREEN, BLUE),
                (0.3, 0.5, 0.6),
                np.linspace(0.0, TAU, 3, endpoint=False),
                strict=True
            )
        ))
        await self.wait(5.0)


class LaggedAnimationExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        text = Text("Text")
        await self.play(Parallel(*(
            Parallel(
                FadeIn(char),
                char.animate(rewind=True).shift(DOWN)
            )
            for char in text
        ), lag_time=0.5), rate=Rates.smooth())
        await self.wait(3.0)


class FormulaExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        factored_formula = MathTex(
            "\\left( a_{0}^{2} + a_{1}^{2} \\right) \\left( b_{0}^{2} + b_{1}^{2} + b_{2}^{2} \\right)",
            local_colors={
                re.compile(r"a_{\d}"): TEAL,
                re.compile(r"b_{\d}"): ORANGE
            }
        ).scale(0.5).shift(UP)
        expanded_formula = MathTex(
            "a_{0}^{2} b_{0}^{2} + a_{0}^{2} b_{1}^{2} + a_{0}^{2} b_{2}^{2}" \
                + " + a_{1}^{2} b_{0}^{2} + a_{1}^{2} b_{1}^{2} + a_{1}^{2} b_{2}^{2}",
            local_colors={
                re.compile(r"a_{\d}"): TEAL,
                re.compile(r"b_{\d}"): ORANGE
            }
        ).scale(0.5).shift(DOWN)
        self.add(factored_formula)
        await self.wait()
        await self.play(TransformMatchingStrings(factored_formula, expanded_formula), rate=Rates.smooth(), run_time=2.0)
        await self.wait(2.0)


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
        timelines = [
            square.animate().shift(UP)
            for square in squares
        ]
        for timeline in timelines:
            self.prepare(
                timeline,
                rate=Rates.smooth(),
                launch_condition=KeyPress(KEY.SPACE).captured
            )
        await self.wait_until(lambda: all(timeline.terminated() for timeline in timelines))
        await self.wait()


class NoteTimeline(Timeline):
    __slots__ = (
        "_note",
        "_key"
    )

    def __init__(
        self,
        note: Mobject,
        key: int
    ) -> None:
        super().__init__()
        self._note: Mobject = note
        self._key: int = key

    async def construct(
        self: Self
    ) -> None:
        note = self._note
        self.scene.add(note)
        event = KeyPress(self._key)
        await self.play(
            note.animate(infinite=True).shift(7.0 * DOWN),
            terminate_condition=lambda: event.captured() and -3.4 <= note.box.get()[1] <= -2.6 or note.box.get()[1] <= -3.4
        )
        if event.captured():
            await self.play(
                note.animate().set(opacity=0.0).scale(1.5),
                rate=Rates.rush_from()
            )
        else:
            note.set(opacity=0.4)
            await self.play(
                note.animate(infinite=True).shift(10.0 * DOWN),
                terminate_condition=lambda: note.box.get()[1] <= -5.0
            )
        self.scene.discard(note)


class GameExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        score = [
            "|  | |    |      |     |   | |  ",
            " |  |    |     |   |  |   |  | |",
            "| |   |   | |  || |  |  |   | | ",
            "    |  | |    |  |  |   || |   |"
        ]
        keys = [KEY.D, KEY.F, KEY.J, KEY.K]
        x_coords = np.linspace(-3.0, 3.0, 4)
        note_template = (
            Polygon(np.array((
                (0.5, 0.0),
                (0.46, 0.04),
                (-0.46, 0.04),
                (-0.5, 0.0),
                (-0.46, -0.04),
                (0.46, -0.04)
            )))
            .set(
                width=0.25,
                color=BLUE_B,
                opacity=0.95
            )
            .add_strokes()
        )
        judge_line = Line(8.0 * LEFT, 8.0 * RIGHT).shift(3.0 * DOWN).set(
            width=0.03,
            color=GOLD_A
        )
        key_texts = [
            Text(char, concatenate=True).add_strokes().shift(x_coord * RIGHT + 2.0 * DOWN)
            for char, x_coord in zip("DFJK", x_coords, strict=True)
        ]

        self.add(judge_line)
        await self.play(Parallel(*(
            Create(key_text, n_segments=2)
            for key_text in key_texts
        ), lag_time=0.5), run_time=1.5)
        await self.wait()
        await self.play(Parallel(*(
            Uncreate(key_text)
            for key_text in key_texts
        ), lag_time=0.5), run_time=1.5)
        for note_chars in zip(*score, strict=True):
            for note_char, key, x_coord in zip(note_chars, keys, x_coords, strict=True):
                if note_char == " ":
                    continue
                note = note_template.copy().shift(x_coord * RIGHT + 5.0 * UP)
                self.prepare(NoteTimeline(note, key))
            await self.wait(0.25)
        await self.wait(3.0)


def main() -> None:
    with (
        Config(
            #fps=30,
            #pixel_height=540,
        ),
        Toplevel.livestream(),
        #Toplevel.recording("ShapeTransformExample.mp4")
    ):
        ShapeTransformExample().run()


if __name__ == "__main__":
    main()
