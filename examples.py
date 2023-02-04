from manim3 import *


class Demo(Scene):
    def construct(self) -> None:
        text = Text("Example Text")
        self.add(text)
        self.play(Scale(text, RIGHT))
        self.wait()


if __name__ == "__main__":
    Renderer().run(Demo)
