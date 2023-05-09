from typing import Iterator
import numpy as np
from scipy.spatial.transform import Rotation

from manim3 import *


class ShapeTransformExample(Scene):
    def timeline(self) -> Iterator[float]:
        circle = Circle()
        circle.set_style(color=Palette.PINK, opacity=0.9)
        circle.add(circle.build_stroke(color=Palette.YELLOW, width=0.4))
        square = Square()
        square.set_style(opacity=1.0)

        self.add(square)
        yield from self.play(Transform(square, circle))
        yield from self.wait()


class TexTransformExample(Scene):
    def timeline(self) -> Iterator[float]:
        text = (
            Text("Text")
            .scale(3)
            .set_style(color=Palette.ORANGE, opacity=0.5)
            .concatenate()
        )
        text.add(
            text.build_stroke(width=0.04, color=Palette.BLUE),
            text.build_stroke(width=0.08, color=Palette.GREEN)
        )
        tex = (
            Tex("Tex")
            .scale(3)
            .set_style(color=Palette.BLUE, opacity=0.5)
            .concatenate()
            .shift(RIGHT * 2)
        )
        tex.add(tex.build_stroke(width=0.06, color=Palette.PINK))
        self.add(text)
        yield from self.wait()
        yield from self.play(Transform(text, tex))
        yield from self.wait(1)
        tex_copy = tex.copy().shift(RIGHT * 2)
        yield from self.play(Transform(tex, tex_copy))
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
    def timeline(self) -> Iterator[float]:
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
    def timeline(self) -> Iterator[float]:
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
    ThreeDTextExample.render(config)


if __name__ == "__main__":
    main()
