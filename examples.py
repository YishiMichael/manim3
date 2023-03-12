from manim3 import *


class TextExample(Scene):
    def construct(self) -> None:
        text = Text("Example Text")
        target = text.copy().shift(RIGHT * 3).set_fill(color="#00ff00")
        for shape, target_shape in zip(text.iter_children(), target.iter_children()):
            self.add(shape)
            if isinstance(shape, ShapeMobject) and isinstance(target_shape, ShapeMobject):
                self.prepare(Transform(shape, target_shape))


class ShapeTransformExample(Scene):
    def construct(self) -> None:
        circle = Circle()
        circle.set_fill(color=BLUE, opacity=0.5)
        circle.set_stroke(color=BLUE_E, width=4)
        square = Square()

        #self.play(ShowCreation(square))
        self.add(square)
        self.wait()
        self.play(Transform(square, circle))
        self.wait(5)


class TexTransformExample(Scene):
    def construct(self) -> None:
        text = Text("Text").scale(3).add_stroke(width=0.1, color=RED).add_stroke(width=0.5, color=YELLOW).concatenate()
        tex = TexText("TexText").scale(3).set_fill(color=BLUE).set_stroke(width=0.3, color=PINK).concatenate()
        self.add(text)
        self.wait()
        self.play(Transform(text, tex.shift(RIGHT * 2)))
        self.wait()


if __name__ == "__main__":
    config = Config()
    #config.fps = 3
    #config.preview = False
    #config.write_video = True
    #config.window_pixel_size = (1920, 1080)
    Renderer(config).run(TexTransformExample)
