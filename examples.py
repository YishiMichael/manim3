from manim3 import *


class Demo(Scene):
    def construct(self) -> None:
        text = Text("Example Text")
        self.add(text)
        self.play(ShapeMobjectTransform(text, text.copy().shift(RIGHT)))
        self.wait()


if __name__ == "__main__":
    Renderer().run(Demo)
