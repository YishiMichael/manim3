import numpy as np

from manim3 import *


class ShapeTransformExample(Scene):
    async def timeline(self) -> None:
        circle = (
            Circle()
            .set_style(color=Palette.PINK, opacity=0.9)
        )
        circle.add(
            circle.build_stroke()
            .set_style(color=Palette.YELLOW, width=0.4)
        )
        square = (
            Square()
            .set_style(opacity=1.0)
        )
        square.add(
            square.build_stroke()
            .set_style(color=Palette.YELLOW, width=0.0)
        )

        self.add(square)
        await self.wait()
        await self.play(Transform(square, circle, run_time=2, rate_func=RateUtils.smooth))
        await self.wait()


class TexTransformExample(Scene):
    async def timeline(self) -> None:
        text = (
            Text("Text")
            .scale(3)
            .set_style(color=Palette.ORANGE, opacity=0.5)
            .concatenate()
        )
        text.add(
            text.build_stroke()
            .set_style(color=Palette.BLUE, width=0.04)
        )
        tex = (
            Tex("Tex")
            .scale(3)
            .set_style(color=Palette.BLUE, opacity=0.5)
            .concatenate()
            .shift(RIGHT * 2)
        )
        tex.add(
            tex.build_stroke()
            .set_style(color=Palette.PINK, width=0.06)
        )
        self.add(text)
        await self.wait()
        await self.play(Transform(text, tex, run_time=2, rate_func=RateUtils.smooth))
        await self.wait()
        await self.play(TransformTo(tex, tex.copy().shift(RIGHT * 2), run_time=2, rate_func=RateUtils.smooth))
        await self.wait(3)


class CreateTexExample(Scene):
    async def timeline(self) -> None:
        text = (
            Text("Text")
            .scale(3)
            .set_style(color=Palette.ORANGE, opacity=0.5)
            .concatenate()
        )
        text.add(
            text.build_stroke()
            .set_style(color=Palette.BLUE, width=0.04)
        )
        await self.wait()
        await self.play(PartialCreate(text, run_time=2, rate_func=RateUtils.smooth))
        await self.wait()
        await self.play(PartialUncreate(text, run_time=2, rate_func=RateUtils.smooth, backwards=True))
        await self.wait()


class ThreeDTextExample(Scene):
    async def timeline(self) -> None:
        text = Text("Text").concatenate()
        text_3d = (
            MeshMobject()
            .set_style(geometry=PrismoidGeometry(text._shape_))
            .scale(5.0)
            .scale_to(0.5, alpha=Z_AXIS)
            .set_style(color="#00FFAA44", is_transparent=True)
        )
        self.add(AmbientLight().set_style(opacity=0.3))
        self.add(PointLight().shift(RIGHT * 5))
        self.add(text_3d)
        self.prepare(Rotating(text_3d, 0.5 * DOWN))
        await self.wait(10)


class OITExample(Scene):
    async def timeline(self) -> None:
        self.add(*(
            (Circle()
                .set_style(color=color, opacity=opacity, is_transparent=True)
                .shift(RIGHT * 0.5)
                .rotate(OUT * angle)
            )
            for color, opacity, angle in zip(
                (Palette.RED, Palette.GREEN, Palette.BLUE),
                (0.3, 0.5, 0.6),
                np.linspace(0, TAU, 3, endpoint=False)
            )
        ))
        await self.wait(5)


class ChildSceneExample(Scene):
    async def timeline(self) -> None:
        child_scene_1 = ThreeDTextExample()
        child_scene_1.render_passes.append(PixelatedPass())
        self.prepare(child_scene_1)
        self.add(
            ChildSceneMobject(child_scene_1)
            .scale(0.5)
            .shift(LEFT * 1)
        )
        child_scene_2 = TexTransformExample()
        self.prepare(child_scene_2)
        self.add(
            ChildSceneMobject(child_scene_2)
            .scale(0.5)
            .shift(RIGHT * 1)
            .shift(OUT * 0.01)
            .set_style(is_transparent=True)
        )
        await self.wait(6)


class LaggedAnimationExample(Scene):
    async def timeline(self) -> None:
        text = Text("Text").scale(3).set_style(opacity=1.0)
        await self.play(LaggedParallel(*(
            Parallel(
                FadeIn(char),
                Shift(char, UP, arrive=True)
            )
            for char in text
        ), lag_time=0.4, rate_func=RateUtils.smooth))
        await self.wait()


class FormulaExample(Scene):
    async def timeline(self) -> None:
        explicit_formula = Tex(
            "\\int_{0}^{\\infty} \\mathrm{e}^{- t}"
                + " \\left( c_{0} + c_{1} t + c_{2} t^{2} + \\cdots + c_{n} t^{n} \\right) \\mathrm{d} t",
            isolate=[
                "\\int_{0}^{\\infty} \\mathrm{e}^{- t}",
                "\\mathrm{d} t",
                "c_{0}",
                "c_{1} t",
                "c_{2} t^{2}",
                "c_{n} t^{n}"
            ],
            tex_to_color_map={
                "\\mathrm{e}": Palette.MAROON_A,
                "c_{0}": Palette.BLUE,
                "c_{1}": Palette.BLUE,
                "c_{2}": Palette.BLUE,
                "c_{n}": Palette.BLUE
            }
        ).scale(0.7)
        expanded_formula = Tex(
            "\\int_{0}^{\\infty} \\mathrm{e}^{- t} c_{0} \\mathrm{d} t"
                + " + \\int_{0}^{\\infty} \\mathrm{e}^{- t} c_{1} t \\mathrm{d} t"
                + " + \\int_{0}^{\\infty} \\mathrm{e}^{- t} c_{2} t^{2} \\mathrm{d} t"
                + " + \\cdots"
                + " + \\int_{0}^{\\infty} \\mathrm{e}^{- t} c_{n} t^{n} \\mathrm{d} t",
            isolate=[
                "\\int_{0}^{\\infty} \\mathrm{e}^{- t}",
                "\\mathrm{d} t",
                "c_{0}",
                "c_{1} t",
                "c_{2} t^{2}",
                "c_{n} t^{n}"
            ],
            tex_to_color_map={
                "\\mathrm{e}": Palette.MAROON_A,
                "c_{0}": Palette.BLUE,
                "c_{1}": Palette.BLUE,
                "c_{2}": Palette.BLUE,
                "c_{n}": Palette.BLUE
            }
        ).scale(0.7)
        self.add(explicit_formula)
        await self.wait()
        await self.play(TransformMatchingStrings(explicit_formula, expanded_formula, run_time=5, rate_func=RateUtils.smooth))
        await self.wait()


def main() -> None:
    config = Config()
    #config.tex.use_mathjax = True
    #config.rendering.time_span = (2.0, 3.0)
    #config.rendering.fps = 10
    #config.rendering.preview = False
    #config.rendering.write_video = True
    #config.rendering.write_last_frame = True
    #config.size.pixel_size = (960, 540)
    LaggedAnimationExample().render(config)


if __name__ == "__main__":
    main()
