from manim3 import *


class Demo(Scene):
    def construct(self) -> None:
        shape = Text("Example Text")[0]
        target_shape = shape.copy().shift(RIGHT * 3)
        #for shape, target_shape in zip(text.iter_children(), target.iter_children()):
        self.add(shape)
        if isinstance(shape, ShapeMobject) and isinstance(target_shape, ShapeMobject):
            self.prepare(ShapeMobjectTransform(shape, target_shape))
        self.wait(5)


if __name__ == "__main__":
    Renderer().run(Demo)
