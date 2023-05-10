import numpy as np
from scipy.spatial.transform import Rotation

from manim3 import *


class ShapeTransformExample(Scene):
    def timeline(self) -> TimelineT:
        circle = (
            Circle()
            .set_style(color=Palette.PINK, opacity=0.9)
            .add_stroke(color=Palette.YELLOW, width=0.4)
        )
        square = Square()
        square.set_style(opacity=1.0)

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
            .add_stroke(width=0.04, color=Palette.BLUE)
        )
        tex = (
            Tex("Tex")
            .scale(3)
            .set_style(color=Palette.BLUE, opacity=0.5)
            .concatenate()
            .shift(RIGHT * 2)
            .add_stroke(width=0.06, color=Palette.PINK)
        )
        self.add(text)
        yield from self.wait()
        yield from self.play(Transform(text, tex, run_time=2, rate_func=RateUtils.smooth))
        yield from self.wait()
        tex_copy = tex.copy().shift(RIGHT * 2)
        yield from self.play(Transform(tex, tex_copy, run_time=2, rate_func=RateUtils.smooth))
        yield from self.wait(3)


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
        self.scene_state.add_point_light(position=RIGHT)
        text = Text("Text").concatenate()
        text_3d = (
            MeshMobject()
            .set_geometry(PrismoidGeometry(text.get_shape()))
            .scale(5.0)
            .stretch_to_fit_depth(0.5)
            .set_style(color="#00FFAA44")
        )
        self.add(text_3d)
        self.prepare(Rotating(text_3d))
        yield from self.wait(10)


class OITExample(Scene):
    def timeline(self) -> TimelineT:
        self.add(*(
            (Circle()
                .set_style(color=color, opacity=opacity)
                .shift(RIGHT * 0.5)
                .rotate_about_origin(Rotation.from_rotvec(OUT * angle))
            )
            for color, opacity, angle in zip(
                (Palette.RED, Palette.GREEN, Palette.BLUE),
                (0.3, 0.5, 0.6),
                np.linspace(0, TAU, 3, endpoint=False)
            )
        ))
        yield from self.wait(5)


def main():
    config = Config()
    #config.tex.use_mathjax = True
    #config.rendering.time_span = (2.0, 3.0)
    #config.rendering.fps = 3
    #config.rendering.preview = False
    #config.rendering.write_video = True
    #config.size.pixel_size = (960, 540)
    #config.rendering.write_last_frame = True
    TexTransformExample.render(config)


if __name__ == "__main__":
    main()
