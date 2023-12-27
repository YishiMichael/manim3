from __future__ import annotations


import re
from typing import Self

from manim3 import *


class FormulaExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        factored_formula = Math(
            "(a_0 + a_1) (b_0 + b_1 + b_2)"
        ).set_local_colors({
            re.compile(r"a_\d"): TEAL,
            re.compile(r"b_\d"): ORANGE
        }).scale(2.0).shift(UP)
        expanded_formula = Math(
            "a_0 b_0 + a_0 b_1 + a_0 b_2 + a_1 b_0 + a_1 b_1 + a_1 b_2"
        ).set_local_colors({
            re.compile(r"a_\d"): TEAL,
            re.compile(r"b_\d"): ORANGE
        }).scale(2.0).shift(DOWN)
        self.add(factored_formula)
        await self.wait()
        await self.play(FadeTransform(factored_formula, expanded_formula), rate=Rates.smooth(), run_time=2.0)
        await self.wait(2.0)


if __name__ == "__main__":
    with (
        Config(),
        Toplevel.livestream()
    ):
        FormulaExample().run()
