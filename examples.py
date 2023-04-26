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
        yield from self.play(Transform(square, circle, replace=False))
        yield from self.wait()


class TexTransformExample(Scene):
    def timeline(self) -> Iterator[float]:
        text = (Text("Text")
            .scale(3)
            .set_style(color=Palette.ORANGE, opacity=0.5)
            .concatenate()
        )
        text.add(
            text.build_stroke(width=0.04, color=Palette.BLUE),
            text.build_stroke(width=0.08, color=Palette.GREEN)
        )
        tex = (Tex("Tex")
            .scale(3)
            .set_style(color=Palette.BLUE, opacity=0.5)
            .concatenate()
        )
        tex.add(tex.build_stroke(width=0.06, color=Palette.PINK))
        self.add(text)
        yield from self.wait()
        yield from self.play(Transform(text, tex.shift(RIGHT * 2), replace=True))
        yield from self.wait(3)


class Rotating(Animation):
    def __init__(
        self,
        mobject: Mobject
    ) -> None:
        initial_model_matrix = mobject._model_matrix_
        initial_mobject = mobject.copy()

        def updater(
            #alpha_0: float,
            alpha: float
        ) -> None:
            initial_mobject.rotate(Rotation.from_rotvec(DOWN * alpha * 0.5))
            mobject._model_matrix_ = initial_mobject._model_matrix_
            initial_mobject._model_matrix_ = initial_model_matrix
            #mobject.rotate(Rotation.from_rotvec(DOWN * (alpha - alpha_0) * 0.5))

        super().__init__(
            updater=updater
            #alpha_animate_func=alpha_animate_func,
            #alpha_regroup_items=[],
            #start_time=0.0,
            #stop_time=None
        )


class ThreeDTextExample(Scene):
    def timeline(self) -> Iterator[float]:
        self.scene_state.add_point_light(position=4 * RIGHT + 4 * UP + 2 * OUT)
        text = Text("Text").concatenate()
        text_3d = MeshMobject().set_geometry(PrismoidGeometry(text.get_shape()))
        text_3d.scale(5.0).stretch_to_fit_depth(0.5)
        text_3d.set_style(color="#00FFAA44")
        self.add(text_3d)
        self.prepare(Rotating(text_3d))
        yield from self.wait(10)


class OITExample(Scene):
    def timeline(self) -> Iterator[float]:
        self.add(*reversed([
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
        ]))
        yield from self.wait(5)


if __name__ == "__main__":
    config = Config()
    #config.tex.use_mathjax = True
    #config.rendering.time_span = (2.0, 3.0)
    #config.rendering.fps = 3
    #config.rendering.preview = False
    #config.rendering.write_video = True
    #config.size.pixel_size = (960, 540)
    #config.rendering.write_last_frame = True
    TexTransformExample.render(config)
