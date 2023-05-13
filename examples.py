import numpy as np
from scipy.spatial.transform import Rotation

from manim3 import *
from manim3.custom_typing import TimelineT


class ShapeTransformExample(Scene):
    def timeline(self) -> TimelineT:
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
        yield from self.play(Transform(square, circle, run_time=2, rate_func=RateUtils.smooth))
        yield from self.wait()


class TexTransformExample(Scene):
    def timeline(self) -> TimelineT:
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
        yield from self.wait()
        yield from self.play(Transform(text, tex, run_time=2, rate_func=RateUtils.smooth))
        yield from self.wait()
        tex_copy = tex.copy().shift(RIGHT * 2)
        yield from self.play(Transform(tex, tex_copy, run_time=2, rate_func=RateUtils.smooth))
        yield from self.wait(3)


class CreateTexExample(Scene):
    def timeline(self) -> TimelineT:
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
        yield from self.wait()
        self.add(text)
        yield from self.play(PartialCreate(text, run_time=2, rate_func=RateUtils.smooth))
        yield from self.wait()
        yield from self.play(PartialUncreate(text, run_time=2, rate_func=RateUtils.smooth, backwards=True))
        yield from self.wait()


class Rotating(Animation):
    def __init__(
        self,
        mobject: Mobject
    ) -> None:
        initial_model_matrix = mobject._model_matrix_.value
        about_point = mobject.get_bounding_box_point(ORIGIN)

        def updater(
            alpha: float
        ) -> None:
            mobject._model_matrix_ = mobject.get_relative_transform_matrix(
                matrix=SpaceUtils.matrix_from_rotation(Rotation.from_rotvec(DOWN * alpha * 0.5)),
                about_point=about_point
            ) @ initial_model_matrix

        super().__init__(
            updater=updater
        )


class ThreeDTextExample(Scene):
    def timeline(self) -> TimelineT:
        text = Text("Text").concatenate()
        text_3d = (
            MeshMobject()
            .set_style(geometry=PrismoidGeometry(text.shape))
            .scale(5.0)
            .stretch_to_fit_depth(0.5)
            .set_style(color="#00FFAA44")
        )
        self.add(AmbientLight().set_style(opacity=0.3))
        self.add(PointLight().shift(RIGHT * 5))
        self.add(text_3d)
        self.prepare(Rotating(text_3d))
        yield from self.wait(10)


class OITExample(Scene):
    def timeline(self) -> TimelineT:
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
        yield from self.wait(5)


class ChildSceneExample(Scene):
    def timeline(self) -> TimelineT:
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
        yield from self.wait(6)


def main():
    config = Config()
    #config.tex.use_mathjax = True
    #config.rendering.time_span = (2.0, 3.0)
    #config.rendering.fps = 3
    #config.rendering.preview = False
    #config.rendering.write_video = True
    #config.rendering.write_last_frame = True
    #config.size.pixel_size = (960, 540)
    CreateTexExample.render(config)


if __name__ == "__main__":
    main()
