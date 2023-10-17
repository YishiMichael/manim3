from __future__ import annotations


import json
import os
import pathlib
from typing import (
    Self,
    TypedDict,
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


# From `https://www.sublimetext.com/docs/api_reference.html#sublime.View.style_for_scope`.
class _HighlightStyle(TypedDict, total=False):
    foreground: str
    background: str
    bold: bool
    italic: bool
    source_line: int
    source_column: int
    source_file: str


class _Token(TypedDict):
    begin: int
    end: int
    style: _HighlightStyle


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
    def _get_local_span_attrs(
        cls: type[Self],
        input_data: CodeInputT,
        temp_path: pathlib.Path
    ) -> dict[Span, dict[str, str]]:

        def local_config_from_token(
            token: _Token
        ) -> tuple[Span, dict[str, str]]:
            style = token["style"]
            config: dict[str, str] = {}
            if (foreground := style.get("foreground")):
                config["fgcolor"] = foreground
            if (background := style.get("background")):
                config["bgcolor"] = background
            if style.get("bold"):
                config["font_weight"] = "bold"
            if style.get("italic"):
                config["font_style"] = "italic"
            return Span(token["begin"], token["end"]), config

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
                raise IOError("CodeIO: Failed to execute subl command")

            json_str = temp_path.with_suffix(".json").read_text(encoding="utf-8")
            local_span_attrs = dict(
                local_config_from_token(token)
                for token in json.loads(json_str)
            )
        finally:
            for suffix in (language_suffix, ".json"):
                temp_path.with_suffix(suffix).unlink(missing_ok=True)

        #local_span_attrs = super()._get_local_span_attrs(input_data, temp_path)
        for span, local_config in super()._get_local_span_attrs(input_data, temp_path).items():
            local_span_attrs.setdefault(span, {}).update(local_config)
        return local_span_attrs
        #override_local_attrs = super()._get_local_attrs(input_data, temp_path)
        #for span, local_config in override_local_attrs.items():
        #    local_attrs.setdefault(span, {}).update(local_config)
        #return local_attrs


class Code(StringMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[CodeKwargs]
    ) -> None:
        super().__init__(CodeIO.get(CodeInput(string=string, **kwargs)))

    #def __init__(
    #    self: Self,
    #    string: str,
    #    *,
    #    font: str | None = None,
    #    language_suffix: str | None = None,
    #    **kwargs
    #) -> None:
    #    config = Toplevel.config
    #    if font is None:
    #        font = config.code_font
    #    if language_suffix is None:
    #        language_suffix = config.code_language_suffix

    #    super().__init__(
    #        string=string,
    #        font=font,
    #        language_suffix=language_suffix,
    #        **kwargs
    #    )

    #@classmethod
    #@property
    #def _io_cls(
    #    cls: type[Self]
    #) -> type[CodeIO]:
    #    return CodeIO

    #@classmethod
    #@property
    #def _input_data_cls(
    #    cls: type[Self]
    #) -> type[CodeInput]:
    #    return CodeInput
