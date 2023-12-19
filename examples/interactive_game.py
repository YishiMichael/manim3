from __future__ import annotations


from typing import Self

import numpy as np
from manim3 import *


class NoteTimeline(Timeline):
    __slots__ = (
        "_note",
        "_key"
    )

    def __init__(
        self,
        note: Mobject,
        key: int
    ) -> None:
        super().__init__()
        self._note: Mobject = note
        self._key: int = key

    async def construct(
        self: Self
    ) -> None:
        note = self._note
        self.scene.add(note)
        event = KeyPress(self._key)
        await self.play(
            note.animate(infinite=True).shift(7.0 * DOWN),
            terminate_condition=lambda: event.captured() and -3.4 <= note.box.get()[1] <= -2.6 or note.box.get()[1] <= -3.4
        )
        if event.captured():
            await self.play(
                note.animate().set(opacity=0.0).scale(1.5),
                rate=Rates.rush_from()
            )
        else:
            note.set(opacity=0.4)
            await self.play(
                note.animate(infinite=True).shift(10.0 * DOWN),
                terminate_condition=lambda: note.box.get()[1] <= -5.0
            )
        self.scene.discard(note)


class InteractiveGameExample(Scene):
    async def construct(
        self: Self
    ) -> None:
        score = [
            "|  | |    |      |     |   | |  ",
            " |  |    |     |   |  |   |  | |",
            "| |   |   | |  || |  |  |   | | ",
            "    |  | |    |  |  |   || |   |"
        ]
        keys = [KEY.D, KEY.F, KEY.J, KEY.K]
        x_coords = np.linspace(-3.0, 3.0, 4)
        note_template = (
            Polygon(np.array((
                (0.5, 0.0),
                (0.46, 0.04),
                (-0.46, 0.04),
                (-0.5, 0.0),
                (-0.46, -0.04),
                (0.46, -0.04)
            )))
            .set(
                width=0.25,
                color=BLUE_B,
                opacity=0.95
            )
            .add_strokes()
        )
        judge_line = Line(8.0 * LEFT, 8.0 * RIGHT).shift(3.0 * DOWN).set(
            width=0.03,
            color=GOLD_A
        )
        key_texts = [
            Text(char, concatenate=True).add_strokes().shift(x_coord * RIGHT + 2.0 * DOWN)
            for char, x_coord in zip("DFJK", x_coords, strict=True)
        ]

        self.add(judge_line)
        await self.play(Parallel(*(
            Create(key_text, n_segments=2)
            for key_text in key_texts
        ), lag_time=0.5), run_time=1.5)
        await self.wait()
        await self.play(Parallel(*(
            Uncreate(key_text)
            for key_text in key_texts
        ), lag_time=0.5), run_time=1.5)
        for note_chars in zip(*score, strict=True):
            for note_char, key, x_coord in zip(note_chars, keys, x_coords, strict=True):
                if note_char == " ":
                    continue
                note = note_template.copy().shift(x_coord * RIGHT + 5.0 * UP)
                self.prepare(NoteTimeline(note, key))
            await self.wait(0.25)
        await self.wait(3.0)


if __name__ == "__main__":
    with (
        Config(),
        Toplevel.livestream()
    ):
        InteractiveGameExample().run()
