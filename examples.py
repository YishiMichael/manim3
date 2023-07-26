import re

import manim3 as m3
import numpy as np


class ShapeTransformExample(m3.Scene):
    async def timeline(self) -> None:
        square = (
            m3.Square()
            .set_style(color=m3.WHITE, opacity=1.0)
        )
        square.add(
            square.build_stroke()
            .set_style(color=m3.YELLOW, width=0.0)
        )
        circle = (
            m3.Circle()
            .set_style(color=m3.PINK, opacity=0.9)
        )
        circle.add(
            circle.build_stroke()
            .set_style(color=m3.YELLOW, weight=10)
        )

        self.add(square)
        await self.wait()
        await self.play(m3.Transform(square, circle), run_time=2, rate=m3.Rates.smooth)
        await self.wait()


class TexTransformExample(m3.Scene):
    async def timeline(self) -> None:
        text = (
            m3.Text("Text")
            .scale(3)
            .set_style(color=m3.ORANGE, opacity=0.5)
            .concatenate()
        )
        text.add(
            text.build_stroke()
            .set_style(color=m3.BLUE, weight=10)
        )
        tex = (
            m3.Tex("Tex")
            .scale(3)
            .set_style(color=m3.BLUE, opacity=0.5)
            .concatenate()
            .shift(m3.RIGHT * 2)
        )
        tex.add(
            tex.build_stroke()
            .set_style(color=m3.PINK, weight=10)
        )
        self.add(text)
        await self.wait()
        await self.play(m3.Transform(text, tex), run_time=2, rate=m3.Rates.smooth)
        await self.wait()
        await self.play(m3.TransformTo(tex, tex.copy().shift(m3.RIGHT * 2)), rate=m3.Rates.smooth, run_time=2)
        await self.wait(3)


class CreateTexExample(m3.Scene):
    async def timeline(self) -> None:
        text = (
            m3.Text("Text")
            .scale(3)
            .set_style(color=m3.ORANGE, opacity=0.5)
            .concatenate()
        )
        text.add(
            text.build_stroke()
            .set_style(color=m3.BLUE, weight=10)
        )
        await self.wait()
        await self.play(m3.PartialCreate(text), run_time=2, rate=m3.Rates.smooth)
        await self.wait()
        await self.play(m3.PartialUncreate(text, backwards=True), rate=m3.Rates.smooth, run_time=2)
        await self.wait()


class ThreeDTextExample(m3.Scene):
    async def timeline(self) -> None:
        text = m3.Text("Text").concatenate()
        text_3d = (
            m3.MeshMobject(m3.PrismoidMesh(text._shape_))
            .scale(5.0)
            .scale_to(0.5, alpha=m3.Z_AXIS)
            .set_style(
                color="#00FFAA",
                opacity=0.25,
                lighting=m3.Lighting(
                    m3.AmbientLight().set_style(color=m3.WHITE * 0.3),
                    m3.PointLight().shift(m3.RIGHT * 5)
                )
            )
        )
        self.add(text_3d)
        self.prepare(m3.Rotating(self.camera, 0.5 * m3.DOWN))
        await self.wait(10)


class OITExample(m3.Scene):
    async def timeline(self) -> None:
        self.add(*(
            (m3.Circle()
                .set_style(color=color, opacity=opacity)
                .shift(m3.RIGHT * 0.5)
                .rotate(m3.OUT * angle)
            )
            for color, opacity, angle in zip(
                (m3.RED, m3.GREEN, m3.BLUE),
                (0.3, 0.5, 0.6),
                np.linspace(0, m3.TAU, 3, endpoint=False)
            )
        ))
        await self.wait(5)


class LaggedAnimationExample(m3.Scene):
    async def timeline(self) -> None:
        text = m3.Text("Text").scale(3)
        await self.play(m3.Parallel(*(
            m3.Parallel(
                m3.FadeIn(char),
                m3.Shift(char, m3.UP, arrive=True)
            )
            for char in text
        ), lag_ratio=0.5), rate=m3.Rates.smooth, run_time=2.2)
        await self.wait()


class FormulaExample(m3.Scene):
    async def timeline(self) -> None:
        factored_formula = m3.Tex(
            "\\left( a_{0}^{2} + a_{1}^{2} \\right) \\left( b_{0}^{2} + b_{1}^{2} + b_{2}^{2} \\right)",
            tex_to_color_map={
                re.compile(r"a_{\d}"): m3.TEAL,
                re.compile(r"b_{\d}"): m3.ORANGE
            }
        ).scale(0.7)
        expanded_formula = m3.Tex(
            "a_{0}^{2} b_{0}^{2} + a_{0}^{2} b_{1}^{2} + a_{0}^{2} b_{2}^{2}" \
                + " + a_{1}^{2} b_{0}^{2} + a_{1}^{2} b_{1}^{2} + a_{1}^{2} b_{2}^{2}",
            tex_to_color_map={
                re.compile(r"a_{\d}"): m3.TEAL,
                re.compile(r"b_{\d}"): m3.ORANGE
            }
        ).scale(0.7)
        self.add(factored_formula)
        await self.wait()
        await self.play(m3.TransformMatchingStrings(factored_formula, expanded_formula), rate=m3.Rates.smooth, run_time=2)
        await self.wait()


def main() -> None:
    config = m3.Config(
        fps=30,
        #preview=False,
        #tex_use_mathjax=True,
        #write_video=True,
        #write_last_frame=True,
        #pixel_height=480
    )
    TexTransformExample.render(config)


if __name__ == "__main__":
    main()
