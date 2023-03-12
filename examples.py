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
        #self.wait()
        text = Text("Text").scale(3)
        text_concatenated = ShapeMobject(Shape.concatenate(
            glyph._shape_ for glyph in text._shape_mobjects_
        )).apply_transform(text._shape_mobjects_[0]._model_matrix_.value)
        tex = TexText("TexText").scale(3)
        tex_concatenated = ShapeMobject(Shape.concatenate(
            glyph._shape_ for glyph in tex._shape_mobjects_
        )).apply_transform(tex._shape_mobjects_[0]._model_matrix_.value).set_fill(color=BLUE).set_stroke(width=0.1, color=RED)
        #tex_concatenated.next_to(text, DOWN)
        #text = Text("Text").scale(3)
        #text_concatenated = ShapeMobject(Shape.concatenate(
        #    glyph._shape_ for glyph in text._shape_mobjects
        #)).apply_transform(text._shape_mobjects[0]._model_matrix_).set_fill(color=PINK).set_stroke(width=0.1, color=GREEN)
        #self.wait(2)
        self.add(text_concatenated)
        self.wait()
        #tex_concatenated.shift(RIGHT * 3)
        #tex_concatenated.shift(RIGHT * 2)
        #tex_concatenated.remove(*tex_concatenated._stroke_mobjects_)
        #self.wait(3)
        #self.add(tex_concatenated)
        self.play(Transform(text_concatenated, tex_concatenated.shift(RIGHT * 3)))
        self.wait()


if __name__ == "__main__":
    config = Config()
    #config.fps = 3
    #config.preview = False
    #config.write_video = True
    #config.window_pixel_size = (1920, 1080)
    Renderer(config).run(TexTransformExample)
