from __future__ import annotations


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


@attrs.frozen(kw_only=True)
class CodeInputs(TypstMobjectInputs):
    syntax: str = attrs.field(factory=lambda: Toplevel._get_config().code_syntax)


class Code(TypstMobject[CodeInputs]):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[CodeKwargs],
    ) -> None:
        super().__init__(CodeInputs(string=string, **kwargs))

    @classmethod
    def _get_environment_pair_from_inputs(
        cls: type[Self],
        inputs: CodeInputs
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
