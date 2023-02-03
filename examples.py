from manim3 import *


class Demo(Scene):
    def construct(self) -> None:
        text = Text("Example Text")
        self.add(text)
        self.play(Shift(text, RIGHT))


if __name__ == "__main__":
    Renderer().run(Demo)
