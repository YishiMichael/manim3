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

from ...constants.custom_typing import (
    AlignmentType,
    ColorType
)
from ...toplevel.toplevel import Toplevel
from ...utils.color_utils import ColorUtils
from .string_mobject import (
    CommandInfo,
    StandaloneCommandInfo,
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
    color: ColorType = attrs.field(factory=lambda: Toplevel.config.default_color)
    font_size: float = attrs.field(factory=lambda: Toplevel.config.pango_font_size)
    alignment: AlignmentType = attrs.field(factory=lambda: Toplevel.config.pango_alignment)
    font: str = attrs.field(factory=lambda: Toplevel.config.pango_font)
    justify: bool = attrs.field(factory=lambda: Toplevel.config.pango_justify)
    indent: float = attrs.field(factory=lambda: Toplevel.config.pango_indent)
    line_width: float = attrs.field(factory=lambda: Toplevel.config.pango_line_width)


class PangoStringMobjectKwargs(StringMobjectKwargs, total=False):
    color: ColorType
    font_size: float
    alignment: AlignmentType
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

    @classmethod
    def _get_global_span_attribs(
        cls: type[Self],
        input_data: PangoStringMobjectInputT,
        temp_path: pathlib.Path
    ) -> dict[str, str]:
        global_span_attribs = {
            "foreground": ColorUtils.color_to_hex(input_data.color),
            "font_size": str(round(input_data.font_size * 1024.0)),
            "font_family": input_data.font
        }
        global_span_attribs.update(super()._get_global_span_attribs(input_data, temp_path))
        return global_span_attribs

    @classmethod
    def _create_svg(
        cls: type[Self],
        content: str,
        input_data: PangoStringMobjectInputT,
        svg_path: pathlib.Path
    ) -> None:
        if MarkupUtils is None:
            raise OSError("PangoStringMobjectIO: manimpango is not found")
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
    def _get_command_pair(
        cls: type[Self],
        attribs: dict[str, str]
    ) -> tuple[str, str]:
        return f"<span {" ".join(
            f"{key}='{value}'"
            for key, value in attribs.items()
        )}>", "</span>"

    @classmethod
    def _convert_attribs_for_labelling(
        cls: type[Self],
        attribs: dict[str, str],
        label: int | None
    ) -> dict[str, str]:

        def convert_attrib_value(
            key: str,
            value: str
        ) -> str | None:
            if key in (
                "foreground",
                "fgcolor",
                "color"
            ):
                return None
            if key in (
                "background",
                "bgcolor",
                "underline_color",
                "overline_color",
                "strikethrough_color"
            ):
                return "black"
            return value

        result = {
            key: converted_value
            for key, value in attribs.items()
            if (converted_value := convert_attrib_value(key, value)) is not None
        }
        if label is not None:
            result["foreground"] = f"#{label:06x}"
        return result

    @classmethod
    def _iter_command_infos(
        cls: type[Self],
        string: str
    ) -> Iterator[CommandInfo]:
        pattern = re.compile(r"""[<>&"']""")
        for match in pattern.finditer(string):
            yield StandaloneCommandInfo(match, replacement=cls._markup_escape(match.group()))

    @classmethod
    def _markup_escape(
        cls: type[Self],
        substr: str
    ) -> str:
        return cls._MARKUP_ESCAPE_DICT.get(substr, substr)
