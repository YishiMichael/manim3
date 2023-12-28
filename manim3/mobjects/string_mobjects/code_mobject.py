from __future__ import annotations
import itertools


import pathlib
from typing import (
    Self,
    Unpack
)

import attrs

from ...constants.custom_typing import SelectorType
from ...toplevel.toplevel import Toplevel
from .text_mobject import Text
from .typst_mobject import (
    TypstMobject,
    TypstMobjectInputs,
    TypstMobjectKwargs
)


class CodeKwargs(TypstMobjectKwargs, total=False):
    syntax: str
    theme: str | pathlib.Path | None


@attrs.frozen(kw_only=True)
class CodeInputs(TypstMobjectInputs):
    syntax: str = attrs.field(
        factory=lambda: Toplevel._get_config().code_syntax
    )
    theme: str | pathlib.Path | None = attrs.field(
        factory=lambda: Toplevel._get_config().code_theme
    )


class Code(TypstMobject[CodeInputs]):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[CodeKwargs],
    ) -> None:
        super().__init__(CodeInputs(string=string, **kwargs))

    @classmethod
    def _get_preamble_from_inputs(
        cls: type[Self],
        inputs: CodeInputs,
        temp_path: pathlib.Path
    ) -> str:
        return "\n".join(filter(None, (
            super()._get_preamble_from_inputs(inputs, temp_path),
            f"""#set raw(theme: "{
                pathlib.Path("/".join(itertools.repeat("..", len(temp_path.parts) - 1))).joinpath(inputs.theme)
            }")""" if inputs.theme is not None else ""
        )))

    @classmethod
    def _get_environment_pair_from_inputs(
        cls: type[Self],
        inputs: CodeInputs,
        temp_path: pathlib.Path
    ) -> tuple[str, str]:
        return f"```{inputs.syntax}\n", "\n```"

    @classmethod
    def _get_labelled_inputs(
        cls: type[Self],
        inputs: CodeInputs,
        label_to_selector_dict: dict[int, SelectorType]
    ) -> CodeInputs:
        return CodeInputs(
            string=inputs.string,
            preamble="\n".join((inputs.preamble, *(
                Text._format_selector_show_command(selector, label)
                for label, selector in label_to_selector_dict.items()
            ))),
            concatenate=inputs.concatenate,
            align=inputs.align,
            font=inputs.font,
            color=inputs.color,
            syntax="txt"
        )
