import numpy as np
import re

from manim3 import *


class ShapeTransformExample(Scene):
    async def timeline(self) -> None:
        square = (
            Square()
            .set_style(color=WHITE, opacity=1.0)
        )
        square.add(
            square.build_stroke()
            .set_style(color=YELLOW, width=0.0)
        )
        circle = (
            Circle()
            .set_style(color=PINK, opacity=0.9)
        )
        circle.add(
            circle.build_stroke()
            .set_style(color=YELLOW, weight=10)
        )

        self.add(square)
        await self.wait()
        await self.play(Transform(square, circle), run_time=2, rate=Rates.smooth)
        await self.wait()


class TexTransformExample(Scene):
    async def timeline(self) -> None:
        text = (
            #ShapeMobject(Shape(Stroke(
            #    points=np.array([
            #        [0,1,0],
            #        [0,0,0],
            #        [1,0,0],
            #        [1,1,0],
            #        [2,1,0],
            #        [2,0,0],
            #        [3,0,0]
            #    ], dtype=np.float64),
            #    disjoints=np.array([4])
            #)))
            Text("Text")
            .scale(3)
            .set_style(color=ORANGE, opacity=0.5)
            .concatenate()
        )
        text.add(
            text.build_stroke()
            .set_style(color=BLUE, weight=10)
        )
        tex = (
            #ShapeMobject(Shape(Stroke(
            #    points=np.array([
            #        [-3,0.5,0],
            #        [-3,0,0],
            #        [-1,0,0],
            #        [0,0,0],
            #        [0,-2,0],
            #        [2,0,0],
            #        [2,1.5,0]
            #    ]),
            #    disjoints=np.array([2, 5])
            #)))
            Tex("Tex")
            .scale(3)
            .set_style(color=BLUE, opacity=0.5)
            .concatenate()
            .shift(RIGHT * 2)
        )
        tex.add(
            tex.build_stroke()
            .set_style(color=PINK, weight=10)
        )
        self.add(text)
        await self.wait()
        await self.play(Transform(text, tex), run_time=2, rate=Rates.smooth)
        await self.wait()
        await self.play(TransformTo(tex, tex.copy().shift(RIGHT * 2)), rate=Rates.smooth, run_time=2)
        await self.wait(3)


class CreateTexExample(Scene):
    async def timeline(self) -> None:
        text = (
            Text("Text")
            .scale(3)
            .set_style(color=ORANGE, opacity=0.5)
            .concatenate()
        )
        text.add(
            text.build_stroke()
            .set_style(color=BLUE, weight=10)
        )
        await self.wait()
        await self.play(PartialCreate(text), run_time=2, rate=Rates.smooth)
        await self.wait()
        await self.play(PartialUncreate(text, backwards=True), rate=Rates.smooth, run_time=2)
        await self.wait()


class ThreeDTextExample(Scene):
    async def timeline(self) -> None:
        text = Text("Text").concatenate()
        text_3d = (
            MeshMobject()
            .set_style(geometry=PrismoidGeometry(text._shape_))
            .scale(5.0)
            .scale_to(0.5, alpha=Z_AXIS)
            .set_style(
                color="#00FFAA",
                opacity=0.25,
                lighting=Lighting(
                    AmbientLight().set_style(color=WHITE * 0.3),
                    PointLight().shift(RIGHT * 5)
                )
            )
        )
        self.add(text_3d)
        self.prepare(Rotating(self.camera, 0.5 * DOWN))
        await self.wait(10)


class OITExample(Scene):
    async def timeline(self) -> None:
        self.add(*(
            (Circle()
                .set_style(color=color, opacity=opacity)
                .shift(RIGHT * 0.5)
                .rotate(OUT * angle)
            )
            for color, opacity, angle in zip(
                (RED, GREEN, BLUE),
                (0.3, 0.5, 0.6),
                np.linspace(0, TAU, 3, endpoint=False)
            )
        ))
        await self.wait(5)


class LaggedAnimationExample(Scene):
    async def timeline(self) -> None:
        text = Text("Text").scale(3)
        await self.play(Parallel(*(
            Parallel(
                FadeIn(char),
                Shift(char, UP, arrive=True)
            )
            for char in text
        ), lag_ratio=0.5), rate=Rates.smooth, run_time=2.2)
        await self.wait()


class FormulaExample(Scene):
    async def timeline(self) -> None:
        factored_formula = Tex(
            "\\left( a_{0}^{2} + a_{1}^{2} \\right) \\left( b_{0}^{2} + b_{1}^{2} + b_{2}^{2} \\right)",
            tex_to_color_map={
                re.compile(r"a_{\d}"): TEAL,
                re.compile(r"b_{\d}"): ORANGE
            }
        ).scale(0.7)
        expanded_formula = Tex(
            "a_{0}^{2} b_{0}^{2} + a_{0}^{2} b_{1}^{2} + a_{0}^{2} b_{2}^{2}" \
                + " + a_{1}^{2} b_{0}^{2} + a_{1}^{2} b_{1}^{2} + a_{1}^{2} b_{2}^{2}",
            tex_to_color_map={
                re.compile(r"a_{\d}"): TEAL,
                re.compile(r"b_{\d}"): ORANGE
            }
        ).scale(0.7)
        self.add(factored_formula)
        await self.wait()
        await self.play(TransformMatchingStrings(factored_formula, expanded_formula), rate=Rates.smooth, run_time=2)
        await self.wait()


def main() -> None:
    config = Config(
        fps=30,
        #preview=False,
        #tex_use_mathjax=True,
        #write_video = True,
        #write_last_frame=True,
        #pixel_height=480
    )
    TexTransformExample.render(config)


if __name__ == "__main__":
    main()
