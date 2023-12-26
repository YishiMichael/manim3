from __future__ import annotations


from typing import (
    Self,
    Unpack
)

import attrs

from ...constants.custom_typing import SelectorType
from .typst_mobject import (
    TypstMobject,
    TypstMobjectInputs,
    TypstMobjectKwargs
)


class TextKwargs(TypstMobjectKwargs, total=False):
    pass


@attrs.frozen(kw_only=True)
class TextInputs(TypstMobjectInputs):
    pass


class Text(TypstMobject[TextInputs]):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[TextKwargs],
    ) -> None:
        super().__init__(TextInputs(string=string, **kwargs))

    @classmethod
    def _get_labelled_inputs(
        cls: type[Self],
        inputs: TextInputs,
        label_to_selector_dict: dict[int, SelectorType]
    ) -> TextInputs:
        return TextInputs(
            string=inputs.string,
            preamble="\n".join((inputs.preamble, *(
                cls._format_selector_show_command(selector, label)
                for label, selector in label_to_selector_dict.items()
            ))),
            concatenate=inputs.concatenate,
            align=inputs.align,
            font=inputs.font,
            color=inputs.color
        )

    @classmethod
    def _format_selector_show_command(
        cls: type[Self],
        selector: SelectorType,
        label: int
    ) -> str:
        return f"""#show {
            f"\"{selector.replace("\\", "\\\\").replace("\"", "\\\"")}\""
            if isinstance(selector, str)
            else f"regex(\"{selector.pattern.replace("\\", "\\\\").replace("\"", "\\\"")}\")"
        }: set text(fill: rgb(\"#{label:08X}\"))"""
