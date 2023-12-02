from __future__ import annotations


import pathlib
import re
from enum import Enum
from typing import (
    ClassVar,
    Iterator,
    Literal,
    Self,
    TypedDict
)

import attrs

try:
    # Soft dependency.
    from manimpango import MarkupUtils
except ImportError:
    MarkupUtils = None

from ...animatables.arrays.animatable_color import AnimatableColor
from ...constants.custom_typing import (
    AlignmentType,
    ColorType
)
from ...toplevel.toplevel import Toplevel
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


# See `https://docs.gtk.org/Pango/pango_markup.html`.
class PangoAttributes(TypedDict, total=False):
    font: str
    font_desc: str
    font_family: str
    face: str
    font_size: str
    size: str
    font_style: Literal["normal", "oblique", "italic"]
    style: Literal["normal", "oblique", "italic"]
    font_weight: Literal["ultralight", "light", "normal", "bold", "ultrabold", "heavy"] | str
    weight: Literal["ultralight", "light", "normal", "bold", "ultrabold", "heavy"] | str
    font_variant: Literal["normal", "small-caps", "all-small-caps", "petite-caps", "all-petite-caps", "unicase", "title-caps"]
    variant: Literal["normal", "small-caps", "all-small-caps", "petite-caps", "all-petite-caps", "unicase", "title-caps"]
    font_stretch: str
    stretch: str
    font_features: str
    foreground: str
    fgcolor: str
    color: str
    background: str
    bgcolor: str
    alpha: str
    fgalpha: str
    background_alpha: str
    bgalpha: str
    underline: Literal["none", "single", "double", "low", "error"]
    underline_color: str
    overline: Literal["none", "single"]
    overline_color: str
    rise: str
    baseline_shift: str
    font_scale: Literal["superscript", "subscript", "small-caps"]
    strikethrough: Literal["true", "false"]
    strikethrough_color: str
    fallback: Literal["true", "false"]
    lang: str
    letter_spacing: str
    gravity: Literal["south", "east", "north", "west", "auto"]
    gravity_hint: Literal["natural", "strong", "line"]
    show: Literal["none", "spaces", "line-breaks", "spaces|line-breaks", "ignorables", "spaces|ignorables", "line-breaks|ignorables", "spaces|line-breaks|ignorables"]
    insert_hyphens: Literal["true", "false"]
    allow_breaks: Literal["true", "false"]
    line_height: str
    text_transform: Literal["none", "lowercase", "uppercase", "capitalize"]
    segment: Literal["word", "sentence"]


@attrs.frozen(kw_only=True)
class PangoStringMobjectInput(StringMobjectInput[PangoAttributes]):
    color: ColorType = attrs.field(factory=lambda: Toplevel._get_config().default_color)
    alignment: AlignmentType = attrs.field(factory=lambda: Toplevel._get_config().pango_alignment)
    justify: bool = attrs.field(factory=lambda: Toplevel._get_config().pango_justify)
    indent: float = attrs.field(factory=lambda: Toplevel._get_config().pango_indent)
    font: str = attrs.field(factory=lambda: Toplevel._get_config().pango_font)


class PangoStringMobjectKwargs(StringMobjectKwargs[PangoAttributes], total=False):
    color: ColorType
    alignment: AlignmentType
    justify: bool
    indent: float
    font: str


class PangoStringMobjectIO[PangoStringMobjectInputT: PangoStringMobjectInput](StringMobjectIO[PangoAttributes, PangoStringMobjectInputT]):
    __slots__ = ()

    _MARKUP_TAGS: ClassVar[dict[str, PangoAttributes]] = {
        "b": PangoAttributes(font_weight="bold"),
        "big": PangoAttributes(font_size="larger"),
        "i": PangoAttributes(font_style="italic"),
        "s": PangoAttributes(strikethrough="true"),
        "sub": PangoAttributes(baseline_shift="subscript", font_scale="subscript"),
        "sup": PangoAttributes(baseline_shift="superscript", font_scale="superscript"),
        "small": PangoAttributes(font_size="smaller"),
        "tt": PangoAttributes(font_family="monospace"),
        "u": PangoAttributes(underline="single")
    }
    _MARKUP_ESCAPE_DICT: ClassVar[dict[str, str]] = {
        "<": "&lt;",
        ">": "&gt;",
        "&": "&amp;",
        "\"": "&quot;",
        "'": "&apos;"
    }

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
            font="",  # Already handled.
            slant="NORMAL",
            weight="NORMAL",
            size=10,
            _=0,  # Empty parameter.
            disable_liga=False,
            file_name=f"{svg_path}",
            START_X=0,
            START_Y=0,
            width=16384,  # Ensure the canvas is large enough to hold all glyphs.
            height=16384,
            justify=input_data.justify,
            indent=input_data.indent,
            line_spacing=None,  # Already handled.
            alignment=pango_alignment,
            pango_width=None  # No auto wraplines.
        )

    @classmethod
    def _get_adjustment_scale(
        cls: type[Self]
    ) -> float:
        return 0.05626

    @classmethod
    def _iter_global_span_attributes(
        cls: type[Self],
        input_data: PangoStringMobjectInputT,
        temp_path: pathlib.Path
    ) -> Iterator[PangoAttributes]:
        yield PangoAttributes(
            foreground=AnimatableColor._color_to_hex(input_data.color),
            font_family=input_data.font
        )
        yield from super()._iter_global_span_attributes(input_data, temp_path)

    @classmethod
    def _get_empty_attributes(
        cls: type[Self]
    ) -> PangoAttributes:
        return PangoAttributes()

    @classmethod
    def _get_command_pair(
        cls: type[Self],
        attributes: PangoAttributes
    ) -> tuple[str, str]:
        return f"<span {" ".join(
            f"{key}='{value}'"
            for key, value in attributes.items()
        )}>", "</span>"

    @classmethod
    def _convert_attributes_for_labelling(
        cls: type[Self],
        attributes: PangoAttributes,
        label: int | None
    ) -> PangoAttributes:
        result = PangoAttributes(attributes)
        for key in (
            "foreground",
            "fgcolor",
            "color"
        ):
            if key in result:
                result.pop(key)

        for key in (
            "background",
            "bgcolor",
            "underline_color",
            "overline_color",
            "strikethrough_color"
        ):
            if key in result:
                result[key] = "black"

        if label is not None:
            result["foreground"] = f"#{label:06x}"
        return result

    @classmethod
    def _iter_command_infos(
        cls: type[Self],
        string: str
    ) -> Iterator[CommandInfo[PangoAttributes]]:
        pattern = re.compile(r"""[<>&"']""")
        for match in pattern.finditer(string):
            yield StandaloneCommandInfo(match, replacement=cls._markup_escape(match.group()))

    @classmethod
    def _markup_escape(
        cls: type[Self],
        string: str
    ) -> str:
        return cls._MARKUP_ESCAPE_DICT.get(string, string)
