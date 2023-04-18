from scipy.spatial.transform import Rotation

from manim3 import *


class ShapeTransformExample(Scene):
    def construct(self) -> None:
        circle = Circle()
        circle.set_style(color=PINK, opacity=0.9)
        circle.add(circle.build_stroke(color=YELLOW, width=0.4))
        square = Square()
        square.set_style(opacity=1.0)

        self.add(square)
        self.play(Transform(square, circle, replace=False))
        self.wait()


class TexTransformExample(Scene):
    def construct(self) -> None:
        text = (Text("Text")
            .scale(3)
            .set_style(color=ORANGE, opacity=0.5)
            .concatenate()
        )
        text.add(
            text.build_stroke(width=0.04, color=BLUE),
            text.build_stroke(width=0.08, color=GREEN)
        )
        tex = (Tex("Tex")
            .scale(3)
            .set_style(color=BLUE, opacity=0.5)
            .concatenate()
        )
        tex.add(tex.build_stroke(width=0.06, color=PINK))
        #print(len(text._stroke_mobjects_[0]._local_sample_points_.value))
        self.add(text)
        self.wait()
        self.play(Transform(text, tex.shift(RIGHT * 2), replace=True))
        self.wait(3)


class Rotating(Animation):
    def __init__(
        self,
        mobject: Mobject
    ) -> None:

        def alpha_animate_func(
            alpha_0: float,
            alpha: float
        ) -> None:
            mobject.rotate(Rotation.from_rotvec(DOWN * (alpha - alpha_0) * 0.5))

        super().__init__(
            alpha_animate_func=alpha_animate_func,
            alpha_regroup_items=[],
            start_time=0.0,
            stop_time=None
        )


class ThreeDTextExample(Scene):
    def construct(self) -> None:
        self.add_point_light(position=4 * RIGHT + 4 * UP + 2 * OUT)
        text = Text("Text").concatenate()
        text_3d = MeshMobject().set_geometry(PrismoidGeometry(text.get_shape()))
        text_3d.scale(5.0).stretch_to_fit_depth(0.5)
        text_3d.set_style(color="#00FFAA55")
        self.add(text_3d)
        self.prepare(Rotating(text_3d))
        self.wait(10)


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
