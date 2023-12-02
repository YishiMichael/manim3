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


class CreateTexExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        text = (
            Text("Text", concatenate=True)
            .set(color=ORANGE, opacity=0.5)
            .add_strokes(color=BLUE, weight=10.0)
        )
        await self.wait()
        await self.play(Create(text, n_segments=5), rate=Rates.smooth(), run_time=2.0)
        await self.wait()
        await self.play(Uncreate(text, backwards=True, n_segments=5), rate=Rates.smooth(), run_time=2.0)
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
                launch_condition=Events.key_press(KEY.SPACE).captured()
            )
        await self.wait_until(Conditions.all(timeline.terminated() for timeline in timelines))
        await self.wait()


class MobjectPositionInRange(Condition):
    __slots__ = (
        "_mobject",
        "_direction",
        "_x_min",
        "_x_max",
        "_y_min",
        "_y_max",
        "_z_min",
        "_z_max"
    )

    def __init__(
        self,
        mobject: Mobject,
        direction: NP_3f8 = ORIGIN,
        *,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        z_min: float | None = None,
        z_max: float | None = None
    ) -> None:
        super().__init__()
        self._mobject: Mobject = mobject
        self._direction: NP_3f8 = direction
        self._x_min: float | None = x_min
        self._x_max: float | None = x_max
        self._y_min: float | None = y_min
        self._y_max: float | None = y_max
        self._z_min: float | None = z_min
        self._z_max: float | None = z_max

    def judge(self) -> bool:
        position = self._mobject.box.get(self._direction)
        x_val, y_val, z_val = position
        return all((
            (x_min := self._x_min) is None or x_val >= x_min,
            (x_max := self._x_max) is None or x_val <= x_max,
            (y_min := self._y_min) is None or y_val >= y_min,
            (y_max := self._y_max) is None or y_val <= y_max,
            (z_min := self._z_min) is None or z_val >= z_min,
            (z_max := self._z_max) is None or z_val <= z_max
        ))


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
        key_pressed_event = Events.key_press(self._key).captured()
        await self.play(
            note.animate(infinite=True).shift(7.0 * DOWN),
            terminate_condition=Conditions.any((
                Conditions.all((
                    key_pressed_event,
                    MobjectPositionInRange(note, y_min=-3.4, y_max=-2.6)
                )),
                MobjectPositionInRange(note, y_max=-3.4)
            ))
        )
        if key_pressed_event.get_captured_event():
            await self.play(
                note.animate().set(opacity=0.0).scale(1.5),
                rate=Rates.rush_from()
            )
        else:
            note.set(opacity=0.4)
            await self.play(
                note.animate(infinite=True).shift(10.0 * DOWN),
                terminate_condition=MobjectPositionInRange(note, y_max=-5.0)
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
        #Toplevel.recording("WriteExample.mp4")
    ):
        TextTransformExample().run()


if __name__ == "__main__":
    main()
