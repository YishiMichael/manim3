from __future__ import annotations


import itertools
import pathlib
import re
from typing import (
    Self,
    Unpack
)

import attrs

from ...constants.custom_typing import SelectorType
from ...toplevel.toplevel import Toplevel
from .typst_mobject import (
    TypstMobject,
    TypstMobjectInputs,
    TypstMobjectKwargs
)


class MathKwargs(TypstMobjectKwargs, total=False):
    inline: bool


@attrs.frozen(kw_only=True)
class MathInputs(TypstMobjectInputs):
    inline: bool = attrs.field(
        factory=lambda: Toplevel._get_config().math_inline
    )


class Math(TypstMobject[MathInputs]):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[MathKwargs],
    ) -> None:
        super().__init__(MathInputs(string=string, **kwargs))

    @classmethod
    def _get_environment_pair_from_inputs(
        cls: type[Self],
        inputs: MathInputs,
        temp_path: pathlib.Path
    ) -> tuple[str, str]:
        if inputs.inline:
            return "$", "$"
        return "$ ", " $"

    @classmethod
    def _get_labelled_inputs(
        cls: type[Self],
        inputs: MathInputs,
        label_to_selector_dict: dict[int, SelectorType]
    ) -> MathInputs:
        return MathInputs(
            string=cls._insert_commands_around_spans(
                inputs.string,
                tuple(
                    (match.span(), (f"#[#text(rgb(\"#{label:08X}\"))[$", "$]]"))
                    for label, selector in label_to_selector_dict.items()
                    for match in (
                        re.compile(rf"\b{re.escape(selector)}\b" if re.fullmatch(r"[a-zA-Z]]+", selector) else re.escape(selector))
                        if isinstance(selector, str) else selector
                    ).finditer(inputs.string)
                )
            ),
            preamble=inputs.preamble,
            concatenate=inputs.concatenate,
            align=inputs.align,
            font=inputs.font,
            color=inputs.color,
            inline=inputs.inline
        )

    @classmethod
    def _insert_commands_around_spans(
        cls: type[Self],
        string: str,
        insert_pair_items: tuple[tuple[tuple[int, int], tuple[str, str]], ...]
    ) -> str:
        # Assume all spans are non-empty and mutually disjoint. 
        insert_items = sorted(itertools.chain.from_iterable(
            ((start_index, 1, start_command), (stop_index, -1, stop_command))
            for (start_index, stop_index), (start_command, stop_command) in insert_pair_items
        ))
        insert_indices = tuple(index for index, _, _ in insert_items)
        insert_commands = tuple(command for _, _, command in insert_items)
        return "".join(itertools.chain.from_iterable(zip(
            ("", *insert_commands),
            tuple(
                string[start:stop]
                for start, stop in itertools.pairwise((0, *insert_indices, len(string)))
            ),
            strict=True
        )))
