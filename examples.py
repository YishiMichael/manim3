from manim3 import *

import numpy as np


class TextExample(Scene):
    def construct(self) -> None:
        text = Text("Example Text")
        target = text.copy().shift(RIGHT * 3).set_fill(color="#00ff00")
        for shape, target_shape in zip(text.iter_children(), target.iter_children()):
            self.add(shape)
            if isinstance(shape, ShapeMobject) and isinstance(target_shape, ShapeMobject):
                self.prepare(ShapeMobjectTransform(shape, target_shape))


class ShapeTransformExample(Scene):
    def construct(self) -> None:
        circle = Circle()
        circle.set_fill(color=BLUE, opacity=0.5)
        circle.set_stroke(color=BLUE_E, width=4)
        square = Square()

        #self.play(ShowCreation(square))
        self.add(square)
        self.wait()
        self.play(ShapeMobjectTransform(square, circle))
        self.wait(5)


class TexTransformExample(Scene):
    def construct(self) -> None:
        tex = TexText("TexText").scale(3)
        tex_concatenated = ShapeMobject(Shape.concatenate(
            glyph._shape_ for glyph in tex._shape_mobjects
        )).apply_transform(tex._shape_mobjects[0]._model_matrix_)
        text = Text("Text").scale(3)
        text_concatenated = ShapeMobject(Shape.concatenate(
            glyph._shape_ for glyph in text._shape_mobjects
        )).apply_transform(text._shape_mobjects[0]._model_matrix_)
        #tex_concatenated.set_shape(Shape.interpolate_shape(tex_concatenated._shape_, text_concatenated._shape_, 0.01))
        #tex_concatenated = ShapeMobject(Shape(MultiLineString2D([
        #    #LineString2D(np.array([(0,0),(1,0),(1,1),(-1,1),(-1,-2),(1,-2),(1,-1),(0,-1),(0,0)])),
        #    #LineString2D(np.array([(0,0)])),
        #    #LineString2D(np.array([(2,0.5),(3,0.5),(3,2.5),(2,2.5),(2,0.5)])),
        #    #LineString2D(np.array([(2,0.5)]))
        #    LineString2D(np.array([(0,-2),(1,-2),(1,-1),(0,-1),(0,-2)]))
        #])))
        #text_concatenated = ShapeMobject(Shape(MultiLineString2D([
        #    #LineString2D(np.array([(0,0),(1,0),(1,1),(-1,1),(-1,-2),(1,-2),(1,-1),(0,-1),(0,0)])),
        #    #LineString2D(np.array([(1,-1),(0,0)]))
        #    LineString2D(np.array([(0,0),(1,0),(1,1),(0,1),(0,0)])+np.array((-2,0))),
        #    LineString2D(np.array([(0,0),(1,0),(1,1),(0,1),(0,0)])),
        #    LineString2D(np.array([(0,0),(1,0),(1,1),(0,1),(0,0)])+np.array((2,0)))
        #])))
        #text_concatenated.set_shape(Shape.interpolate_shape(tex_concatenated._shape_, text_concatenated._shape_, 0.96))
        self.add(tex_concatenated)
        self.wait()
        self.play(ShapeMobjectTransform(tex_concatenated, text_concatenated, run_time=5))
        self.wait()


if __name__ == "__main__":
    config = Config()
    #config.write_video = True
    #config.pixel_size = (480, 270)
    Renderer(config).run(TexTransformExample)
