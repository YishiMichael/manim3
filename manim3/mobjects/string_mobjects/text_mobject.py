import pathlib
import re
from enum import Enum
from typing import (
    ClassVar,
    Iterable,
    Iterator
)

import manimpango

from ...constants.custom_typing import (
    AlignmentT,
    ColorT,
    SelectorT
)
from ...toplevel.toplevel import Toplevel
from ...utils.color import ColorUtils
from .string_mobject import (
    CommandFlag,
    EdgeFlag,
    StringFileWriter,
    StringMobject,
    StringParser
)


# Ported from `manimpango/enums.pyx`.
class PangoAlignment(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


class PangoUtils:
    __slots__ = ()

    def __new__(cls):
        raise TypeError

    @classmethod
    def validate_markup_string(
        cls,
        markup_str: str
    ) -> None:
        validate_error = manimpango.MarkupUtils.validate(markup_str)
        if not validate_error:
            return
        raise ValueError(
            f"Invalid markup string \"{markup_str}\"\n" +
            f"{validate_error}"
        )

    @classmethod
    def create_markup_svg(
        cls,
        markup_str: str,
        svg_path: pathlib.Path,
        justify: bool,
        indent: float,
        alignment: AlignmentT,
        pango_width: float
    ) -> None:
        # `manimpango` is under construction,
        # so the following code is intended to suit its interface.
        match alignment:
            case "left":
                pango_alignment = PangoAlignment.LEFT
            case "center":
                pango_alignment = PangoAlignment.CENTER
            case "right":
                pango_alignment = PangoAlignment.RIGHT
        manimpango.MarkupUtils.text2svg(
            text=markup_str,
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
            justify=justify,
            indent=indent,
            line_spacing=None,         # Already handled.
            alignment=pango_alignment,
            pango_width=pango_width
        )


class MarkupTextFileWriter(StringFileWriter):
    __slots__ = ()

    _DIR_NAME: ClassVar[str] = "_markup"

    @classmethod
    def create_svg_file(
        cls,
        content: str,
        svg_path: pathlib.Path,
        justify: bool,
        indent: float,
        alignment: AlignmentT,
        line_width: float
    ) -> None:
        PangoUtils.validate_markup_string(content)
        PangoUtils.create_markup_svg(
            markup_str=content,
            svg_path=svg_path,
            justify=justify,
            indent=indent,
            alignment=alignment,
            pango_width=(
                -1 if line_width < 0.0
                else line_width * Toplevel.config.pixel_per_unit
            )
        )


class MarkupTextParser(StringParser):
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
    def _markup_escape(
        cls,
        substr: str
    ) -> str:
        return cls._MARKUP_ESCAPE_DICT.get(substr, substr)

    @classmethod
    def _markup_unescape(
        cls,
        substr: str
    ) -> str:
        return cls._MARKUP_UNESCAPE_DICT.get(substr, substr)

    @classmethod
    def _iter_command_matches(
        cls,
        string: str
    ) -> Iterator[re.Match[str]]:
        pattern = re.compile(r"""
            (?P<tag>
                <
                (?P<close_slash>/)?
                (?P<tag_name>\w+)\s*
                (?P<attr_list>(?:\w+\s*\=\s*(?P<quot>["']).*?(?P=quot)\s*)*)
                (?P<elision_slash>/)?
                >
            )
            |(?P<passthrough>
                <\?.*?\?>|<!--.*?-->|<!\[CDATA\[.*?\]\]>|<!DOCTYPE.*?>
            )
            |(?P<entity>&(?P<unicode>\#(?P<hex>x)?)?(?P<content>.*?);)
            |(?P<char>[>"'])
        """, flags=re.VERBOSE | re.DOTALL)
        yield from pattern.finditer(string)

    @classmethod
    def _get_command_flag(
        cls,
        match_obj: re.Match[str]
    ) -> CommandFlag:
        if match_obj.group("tag"):
            if match_obj.group("close_slash"):
                return CommandFlag.CLOSE
            if not match_obj.group("elision_slash"):
                return CommandFlag.OPEN
        return CommandFlag.OTHER

    @classmethod
    def _replace_for_content(
        cls,
        match_obj: re.Match[str]
    ) -> str:
        if match_obj.group("tag"):
            return ""
        if match_obj.group("char"):
            return cls._markup_escape(match_obj.group("char"))
        return match_obj.group()

    @classmethod
    def _replace_for_matching(
        cls,
        match_obj: re.Match[str]
    ) -> str:
        if match_obj.group("tag") or match_obj.group("passthrough"):
            return ""
        if match_obj.group("entity"):
            if match_obj.group("unicode"):
                base = 10
                if match_obj.group("hex"):
                    base = 16
                return chr(int(match_obj.group("content"), base))
            return cls._markup_unescape(match_obj.group("entity"))
        return match_obj.group()

    @classmethod
    def _get_attrs_from_command_pair(
        cls,
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
        cls,
        label: int | None,
        edge_flag: EdgeFlag,
        attrs: dict[str, str]
    ) -> str:
        if edge_flag == EdgeFlag.STOP:
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


class TextParser(MarkupTextParser):
    __slots__ = ()

    @classmethod
    def _iter_command_matches(
        cls,
        string: str
    ) -> Iterator[re.Match[str]]:
        pattern = re.compile(r"""[<>&"']""")
        yield from pattern.finditer(string)

    @classmethod
    def _get_command_flag(
        cls,
        match_obj: re.Match[str]
    ) -> CommandFlag:
        return CommandFlag.OTHER

    @classmethod
    def _replace_for_content(
        cls,
        match_obj: re.Match[str]
    ) -> str:
        return cls._markup_escape(match_obj.group())

    @classmethod
    def _replace_for_matching(
        cls,
        match_obj: re.Match[str]
    ) -> str:
        return match_obj.group()


class Text(StringMobject):
    __slots__ = ()

    _TEXT_SCALE_FACTOR: ClassVar[float] = 0.01147

    def __init__(
        self,
        string: str,
        *,
        isolate: Iterable[SelectorT] = (),
        protect: Iterable[SelectorT] = (),
        local_configs: dict[SelectorT, dict[str, str]] | None = None,
        global_config: dict[str, str] | None = None,
        justify: bool | None = None,
        indent: float | None = None,
        alignment: AlignmentT | None = None,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str | None = None,
        color: ColorT | None = None,
        markup: bool = False
    ) -> None:
        if markup:
            PangoUtils.validate_markup_string(string)
        if local_configs is None:
            local_configs = {}
        if global_config is None:
            global_config = {}

        config = Toplevel.config
        if justify is None:
            justify = config.text_justify
        if indent is None:
            indent = config.text_indent
        if alignment is None:
            alignment = config.text_alignment
        if line_width is None:
            line_width = config.text_line_width
        if font_size is None:
            font_size = config.text_font_size
        if font is None:
            font = config.text_font
        if color is None:
            color = config.text_color

        global_attrs = {
            "font_size": str(round(font_size * 1024.0)),
            "font_family": font,
            "foreground": ColorUtils.color_to_hex(color)
        }
        global_attrs.update(global_config)

        file_writer = MarkupTextFileWriter(
            justify=justify,
            indent=indent,
            alignment=alignment,
            line_width=line_width
        )
        parser_class = MarkupTextParser if markup else TextParser
        parser = parser_class(
            string=string,
            isolate=isolate,
            protect=protect,
            local_attrs=local_configs,
            global_attrs=global_attrs,
            file_writer=file_writer,
            frame_scale=self._TEXT_SCALE_FACTOR
        )
        super().__init__(
            parser=parser
        )
