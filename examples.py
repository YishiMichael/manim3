from manim3 import *


class Demo(Scene):
    def construct(self) -> None:
        text = Text("Example Text")
        target = text.copy().shift(RIGHT * 3).set_fill(color="#00ff00")
        for shape, target_shape in zip(text.iter_children(), target.iter_children()):
            self.add(shape)
            if isinstance(shape, ShapeMobject) and isinstance(target_shape, ShapeMobject):
                self.prepare(ShapeMobjectTransform(shape, target_shape))
        self.wait(5)


if __name__ == "__main__":
    Renderer().run(Demo)
