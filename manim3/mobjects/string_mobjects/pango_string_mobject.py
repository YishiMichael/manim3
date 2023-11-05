from __future__ import annotations


import pathlib
import re
from enum import Enum
from typing import (
    ClassVar,
    Iterator,
    Self
)

import attrs

try:
    # Soft dependency.
    from manimpango import MarkupUtils
except ImportError:
    MarkupUtils = None

from ...constants.custom_typing import AlignmentT
from ...toplevel.toplevel import Toplevel
from .string_mobject import (
    BoundaryFlag,
    CommandFlag,
    StringMobjectIO,
    StringMobjectInput,
    StringMobjectKwargs
)


# Ported from `manimpango/enums.pyx`.
class PangoAlignment(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


@attrs.frozen(kw_only=True)
class PangoStringMobjectInput(StringMobjectInput):
    font_size: float = attrs.field(factory=lambda: Toplevel.config.pango_font_size)
    alignment: AlignmentT = attrs.field(factory=lambda: Toplevel.config.pango_alignment)
    font: str = attrs.field(factory=lambda: Toplevel.config.pango_font)
    justify: bool = attrs.field(factory=lambda: Toplevel.config.pango_justify)
    indent: float = attrs.field(factory=lambda: Toplevel.config.pango_indent)
    line_width: float = attrs.field(factory=lambda: Toplevel.config.pango_line_width)


class PangoStringMobjectKwargs(StringMobjectKwargs, total=False):
    font_size: float
    alignment: AlignmentT
    font: str
    justify: bool
    indent: float
    line_width: float


class PangoStringMobjectIO[PangoStringMobjectInputT: PangoStringMobjectInput](StringMobjectIO[PangoStringMobjectInputT]):
    __slots__ = ()

    # See `https://docs.gtk.org/Pango/pango_markup.html`.
    _MARKUP_TAGS: ClassVar[dict[str, dict[str, str]]] = {
        "b": {"font_weight": "bold"},
        "big": {"font_size": "larger"},
        "i": {"font_style": "italic"},
        "s": {"strikethrough": "true"},
        "sub": {"baseline_shift": "subscript", "font_scale": "subscript"},
        "sup": {"baseline_shift": "superscript", "font_scale": "superscript"},
        "small": {"font_size": "smaller"},
        "tt": {"font_family": "monospace"},
        "u": {"underline": "single"}
    }
    _MARKUP_ESCAPE_DICT: ClassVar[dict[str, str]] = {
        "<": "&lt;",
        ">": "&gt;",
        "&": "&amp;",
        "\"": "&quot;",
        "'": "&apos;"
    }
    _MARKUP_UNESCAPE_DICT: ClassVar[dict[str, str]] = {
        v: k
        for k, v in _MARKUP_ESCAPE_DICT.items()
    }

    @classmethod
    def _get_global_span_attrs(
        cls: type[Self],
        input_data: PangoStringMobjectInputT,
        temp_path: pathlib.Path
    ) -> dict[str, str]:
        global_span_attrs = {
            "font_size": str(round(input_data.font_size * 1024.0)),
            "font_family": input_data.font
        }
        global_span_attrs.update(super()._get_global_span_attrs(input_data, temp_path))
        return global_span_attrs

    @classmethod
    def _create_svg(
        cls: type[Self],
        content: str,
        input_data: PangoStringMobjectInputT,
        svg_path: pathlib.Path
    ) -> None:
        if MarkupUtils is None:
            raise IOError("PangoStringMobjectIO: manimpango is not found")
        if (validate_error := MarkupUtils.validate(content)):
            raise ValueError(f"Markup error: {validate_error}")

        # `manimpango` is under construction,
        # so the following code is intended to suit its interface.
        match input_data.alignment:
            case "left":
                pango_alignment = PangoAlignment.LEFT
            case "center":
                pango_alignment = PangoAlignment.CENTER
            case "right":
                pango_alignment = PangoAlignment.RIGHT

        MarkupUtils.text2svg(
            text=content,
            font="",                   # Already handled.
            slant="NORMAL",            # Already handled.
            weight="NORMAL",           # Already handled.
            size=1,                    # Already handled.
            _=0,                       # Empty parameter.
            disable_liga=False,
            file_name=str(svg_path),
            START_X=0,
            START_Y=0,
            width=16384,               # Ensure the canvas is large enough
            height=16384,              # to hold all glyphs.
            justify=input_data.justify,
            indent=input_data.indent,
            line_spacing=None,         # Already handled.
            alignment=pango_alignment,
            pango_width=(
                -1 if (line_width := input_data.line_width) < 0.0
                else line_width * Toplevel.config.pixel_per_unit
            )
        )

    @classmethod
    def _get_svg_frame_scale(
        cls: type[Self],
        input_data: PangoStringMobjectInputT
    ) -> float:
        return 0.01147

    @classmethod
    def _iter_command_matches(
        cls: type[Self],
        string: str
    ) -> Iterator[re.Match[str]]:
        pattern = re.compile(r"""[<>&"']""")
        yield from pattern.finditer(string)

    @classmethod
    def _get_command_flag(
        cls: type[Self],
        match_obj: re.Match[str]
    ) -> CommandFlag:
        return CommandFlag.OTHER

    @classmethod
    def _replace_for_content(
        cls: type[Self],
        match_obj: re.Match[str]
    ) -> str:
        return cls._markup_escape(match_obj.group())

    @classmethod
    def _replace_for_matching(
        cls: type[Self],
        match_obj: re.Match[str]
    ) -> str:
        return match_obj.group()

    @classmethod
    def _get_attrs_from_command_pair(
        cls: type[Self],
        open_command: re.Match[str],
        close_command: re.Match[str]
    ) -> dict[str, str] | None:
        pattern = r"""
            (?P<attr_name>\w+)
            \s*\=\s*
            (?P<quot>["'])(?P<attr_val>.*?)(?P=quot)
        """
        tag_name = open_command.group("tag_name")
        if tag_name == "span":
            return {
                match_obj.group("attr_name"): match_obj.group("attr_val")
                for match_obj in re.finditer(
                    pattern, open_command.group("attr_list"), flags=re.VERBOSE | re.DOTALL
                )
            }
        return cls._MARKUP_TAGS.get(tag_name, {})

    @classmethod
    def _get_command_string(
        cls: type[Self],
        label: int | None,
        boundary_flag: BoundaryFlag,
        attrs: dict[str, str]
    ) -> str:
        if boundary_flag == BoundaryFlag.STOP:
            return "</span>"

        converted_attrs = attrs.copy()
        if label is not None:
            for key in (
                "background", "bgcolor",
                "underline_color", "overline_color", "strikethrough_color"
            ):
                if key in converted_attrs:
                    converted_attrs[key] = "black"
            for key in (
                "foreground", "fgcolor", "color"
            ):
                if key in converted_attrs:
                    converted_attrs.pop(key)
            converted_attrs["foreground"] = f"#{label:06x}"
        attrs_str = " ".join(
            f"{key}='{val}'"
            for key, val in converted_attrs.items()
        )
        return f"<span {attrs_str}>"

    @classmethod
    def _markup_escape(
        cls: type[Self],
        substr: str
    ) -> str:
        return cls._MARKUP_ESCAPE_DICT.get(substr, substr)

    @classmethod
    def _markup_unescape(
        cls: type[Self],
        substr: str
    ) -> str:
        return cls._MARKUP_UNESCAPE_DICT.get(substr, substr)
