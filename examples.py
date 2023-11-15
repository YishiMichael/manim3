import re

import numpy as np
from manim3 import *


class ShapeTransformExample(Scene):
    async def construct(self) -> None:
        square = (
            Square()
            .set(color=WHITE, opacity=1.0)
            .add_strokes(color=YELLOW, width=0.0)
        )
        circle = (
            Circle()
            .set(color=PINK, opacity=0.9)
            .add_strokes(color=YELLOW, weight=10)
        )

        self.add(square)
        await self.wait()
        await self.play(Transform(square, circle), run_time=2, rate=Rates.smooth())
        await self.wait()


class TextTransformExample(Scene):
    async def construct(self) -> None:
        text = (
            Text("Text", concatenate=True)
            .scale(3)
            .set(color=ORANGE, opacity=0.5)
            .shift(LEFT * 2)
            .add_strokes(color=BLUE, weight=10)
        )
        tex = (
            Tex("Tex", concatenate=True)
            .scale(3)
            .set(color=BLUE, opacity=0.5)
            .add_strokes(color=PINK, weight=10)
        )
        code = Code("print(\"Code!\")").shift(RIGHT * 2)
        self.add(text)
        await self.wait()
        await self.play(Transform(text, tex), run_time=2, rate=Rates.smooth())
        #await self.wait()
        await self.play(FadeTransform(tex, code), run_time=2, rate=Rates.smooth())
        await self.wait(3)


class CreateTexExample(Scene):
    async def construct(self) -> None:
        text = (
            Text("Text", concatenate=True)
            .scale(3)
            .set(color=ORANGE, opacity=0.5)
            .add_strokes(color=BLUE, weight=10)
        )
        await self.wait()
        await self.play(Create(text, n_segments=5), rate=Rates.smooth(), run_time=2)
        await self.wait()
        await self.play(Uncreate(text, backwards=True, n_segments=5), rate=Rates.smooth(), run_time=2)
        await self.wait()


class ThreeDExample(Scene):
    async def construct(self) -> None:
        dodec = (
            Dodecahedron()
            .scale(2.0)
            .set(
                color="#00FFAA",
                opacity=0.25
            )
            .bind_lighting(Lighting(
                AmbientLight().set(color=WHITE * 0.3),
                PointLight().shift(RIGHT * 5)
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
        await self.wait(3)


class OITExample(Scene):
    async def construct(self) -> None:
        self.add(*(
            (Circle()
                .set(color=color, opacity=opacity)
                .shift(RIGHT * 0.5)
                .rotate_about_origin(OUT * angle)
            )
            for color, opacity, angle in zip(
                (RED, GREEN, BLUE),
                (0.3, 0.5, 0.6),
                np.linspace(0, TAU, 3, endpoint=False)
            )
        ))
        await self.wait(5)


class LaggedAnimationExample(Scene):
    async def construct(self) -> None:
        text = Text("Text").scale(3)
        await self.play(Parallel(*(
            Parallel(
                FadeIn(char),
                char.animate(rewind=True).shift(DOWN)
            )
            for char in text
        ), lag_time=0.5), rate=Rates.smooth())
        await self.wait(3)


class FormulaExample(Scene):
    async def construct(self) -> None:
        factored_formula = MathTex(
            "\\left( a_{0}^{2} + a_{1}^{2} \\right) \\left( b_{0}^{2} + b_{1}^{2} + b_{2}^{2} \\right)",
            local_colors={
                re.compile(r"a_{\d}"): TEAL,
                re.compile(r"b_{\d}"): ORANGE
            }
        ).scale(0.7).shift(UP)
        expanded_formula = MathTex(
            "a_{0}^{2} b_{0}^{2} + a_{0}^{2} b_{1}^{2} + a_{0}^{2} b_{2}^{2}" \
                + " + a_{1}^{2} b_{0}^{2} + a_{1}^{2} b_{1}^{2} + a_{1}^{2} b_{2}^{2}",
            local_colors={
                re.compile(r"a_{\d}"): TEAL,
                re.compile(r"b_{\d}"): ORANGE
            }
        ).scale(0.7).shift(DOWN)
        self.add(factored_formula)
        await self.wait()
        await self.play(TransformMatchingStrings(factored_formula, expanded_formula), rate=Rates.smooth(), run_time=2)
        await self.wait(2)


class InteractiveExample(Scene):
    async def construct(self) -> None:
        squares = ShapeMobject().add(*(
            (
                Circle()
                .scale(0.5)
                .shift(x * RIGHT)
                .set(color=color)
            )
            for x, color in zip(
                np.linspace(-4.0, 4.0, 5),
                (RED, YELLOW, GREEN, BLUE, PURPLE)
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

    async def construct(self) -> None:
        note = self._note
        judge_condition = MobjectPositionInRange(note, y_min=-3.4, y_max=-2.6)
        self.scene.add(note)
        await self.play(
            note.animate(infinite=True).shift(7.0 * DOWN),
            terminate_condition=Conditions.any((
                Conditions.all((
                    Events.key_press(self._key).captured(),
                    judge_condition
                )),
                MobjectPositionInRange(note, y_max=-3.4)
            ))
        )
        if not judge_condition.judge():
            note.set(opacity=0.4)
            await self.play(
                note.animate(infinite=True).shift(10.0 * DOWN),
                terminate_condition=MobjectPositionInRange(note, y_max=-5.0)
            )
            self.scene.discard(note)
            return

        await self.play(Parallel(
            FadeOut(note),
            note.animate().scale(1.5)
        ), rate=Rates.rush_from())


class GameExample(Scene):
    async def construct(self) -> None:
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
    config = Config(
        fps=30,
        #preview=False,
        #write_video=True,
        #write_last_frame=True,
        #pixel_height=540,
    )
    TextTransformExample.render(config)


if __name__ == "__main__":
    main()
