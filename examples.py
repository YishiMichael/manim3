import numpy as np
from scipy.spatial.transform import Rotation

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
        await self.play(ReplacementTransform(square, circle, run_time=2, rate_func=RateUtils.smooth))
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
        await self.play(ReplacementTransform(text, tex, run_time=2, rate_func=RateUtils.smooth))
        await self.wait()
        await self.play(Transform(tex, tex.copy().shift(RIGHT * 2), run_time=2, rate_func=RateUtils.smooth))
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
            .set_style(geometry=PrismoidGeometry(text.shape))
            .scale(5.0)
            .scale_to(0.5, alpha=Z_AXIS)
            .set_style(color="#00FFAA44")
        )
        self.add(AmbientLight().set_style(opacity=0.3))
        self.add(PointLight().shift(RIGHT * 5))
        self.add(text_3d)
        self.prepare(Rotating(text_3d, Rotation.from_rotvec(0.5 * DOWN)))
        await self.wait(10)


class OITExample(Scene):
    async def timeline(self) -> None:
        self.add(*(
            (Circle()
                .set_style(color=color, opacity=opacity)
                .shift(RIGHT * 0.5)
                .rotate(Rotation.from_rotvec(OUT * angle))
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
            .set_style(is_transparent=True)
        )
        child_scene_2 = TexTransformExample()
        self.prepare(child_scene_2)
        self.add(
            ChildSceneMobject(child_scene_2)
            .scale(0.5)
            .shift(RIGHT * 1)
            .set_style(is_transparent=True)
        )
        await self.wait(6)


class LaggedAnimationExample(Scene):
    async def timeline(self) -> None:
        text = Text("Text").scale(3).set_style(opacity=1.0)
        await self.play(LaggedParallel(*(
            Parallel(
                FadeIn(char),
                Shift(char, UP, towards=True)
            )
            for char in text
        ), lag_time=0.4, rate_func=RateUtils.smooth))
        await self.wait()


def main() -> None:
    config = Config()
    #config.tex.use_mathjax = True
    #config.rendering.time_span = (2.0, 3.0)
    #config.rendering.fps = 3
    #config.rendering.preview = False
    #config.rendering.write_video = True
    #config.rendering.write_last_frame = True
    #config.size.pixel_size = (960, 540)
    LaggedAnimationExample.render(config)


if __name__ == "__main__":
    main()
