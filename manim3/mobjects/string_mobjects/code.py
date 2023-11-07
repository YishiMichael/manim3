from __future__ import annotations


import json
import os
import pathlib
from typing import (
    Iterator,
    Self,
    Unpack
)

import attrs

from ...toplevel.toplevel import Toplevel
from .pango_string_mobject import (
    PangoStringMobjectIO,
    PangoStringMobjectInput,
    PangoStringMobjectKwargs
)
from .string_mobject import (
    Span,
    StringMobject
)


@attrs.frozen(kw_only=True)
class CodeInput(PangoStringMobjectInput):
    font: str = attrs.field(factory=lambda: Toplevel.config.code_font)
    language_suffix: str = attrs.field(factory=lambda: Toplevel.config.code_language_suffix)


class CodeKwargs(PangoStringMobjectKwargs, total=False):
    language_suffix: str


class CodeIO[CodeInputT: CodeInput](PangoStringMobjectIO[CodeInputT]):
    __slots__ = ()

    @classmethod
    @property
    def _dir_name(
        cls: type[Self]
    ) -> str:
        return "code"

    @classmethod
    def _iter_local_span_attribs(
        cls: type[Self],
        input_data: CodeInputT,
        temp_path: pathlib.Path
    ) -> Iterator[tuple[Span, dict[str, str]]]:

        yield from super()._iter_local_span_attribs(input_data, temp_path)
        language_suffix = input_data.language_suffix
        try:
            code_path = temp_path.with_suffix(language_suffix)
            code_path.write_text(input_data.string, encoding="utf-8")

            # First open the file, then launch the command.
            # We separate these two steps as file loading is asynchronous,
            # and operations on `view` has no effect while loading.
            if os.system(" ".join((
                "subl",
                f"\"{code_path}\"",
                "--background",   # Don't activate the application.
                ">", os.devnull
            ))) or os.system(" ".join((
                "subl",
                "--background",
                "--command", "export_highlight",
                ">", os.devnull
            ))):
                raise OSError("CodeIO: Failed to execute subl command")

            json_str = temp_path.with_suffix(".json").read_text(encoding="utf-8")
            for token in json.loads(json_str):
                # See `https://www.sublimetext.com/docs/api_reference.html#sublime.View.style_for_scope`.
                style = token["style"]
                attribs: dict[str, str] = {}
                if (foreground := style.get("foreground")):
                    attribs["fgcolor"] = foreground
                if (background := style.get("background")):
                    attribs["bgcolor"] = background
                if style.get("bold"):
                    attribs["font_weight"] = "bold"
                if style.get("italic"):
                    attribs["font_style"] = "italic"
                yield Span(token["begin"], token["end"]), attribs
        finally:
            for suffix in (language_suffix, ".json"):
                temp_path.with_suffix(suffix).unlink(missing_ok=True)


class Code(StringMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[CodeKwargs]
    ) -> None:
        super().__init__(CodeIO.get(CodeInput(string=string, **kwargs)))
