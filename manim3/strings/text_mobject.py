from enum import Enum
import pathlib
import re
from typing import (
    ClassVar,
    Iterator
)

import manimpango
import pygments
import pygments.formatters
import pygments.lexers

from ..config import ConfigSingleton
from ..custom_typing import (
    ColorT,
    SelectorT
)
from ..strings.string_mobject import (
    CommandFlag,
    EdgeFlag,
    StringFileWriter,
    StringMobject,
    StringParser
)
from ..utils.color import ColorUtils


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
        alignment: PangoAlignment,
        pango_width: float
    ) -> None:
        # `manimpango` is under construction,
        # so the following code is intended to suit its interface.
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
            alignment=alignment,
            pango_width=pango_width
        )


class MarkupTextFileWriter(StringFileWriter):
    __slots__ = ()

    _dir_name: ClassVar[str] = "_markup"

    @classmethod
    def create_svg_file(
        cls,
        content: str,
        svg_path: pathlib.Path,
        justify: bool,
        indent: float,
        alignment: PangoAlignment,
        line_width: float | None
    ) -> None:
        PangoUtils.validate_markup_string(content)
        PangoUtils.create_markup_svg(
            markup_str=content,
            svg_path=svg_path,
            justify=justify,
            indent=indent,
            alignment=alignment,
            pango_width=(
                -1 if line_width is None
                else line_width * ConfigSingleton().size.pixel_per_unit
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
    _MARKUP_ENTITY_DICT: ClassVar[dict[str, str]] = {
        "<": "&lt;",
        ">": "&gt;",
        "&": "&amp;",
        "\"": "&quot;",
        "'": "&apos;"
    }
    _MARKUP_ENTITY_REVERSED_DICT: ClassVar[dict[str, str]] = {
        v: k
        for k, v in _MARKUP_ENTITY_DICT.items()
    }

    #def __init__(
    #    self,
    #    string: str,
    #    isolate: SelectorT,
    #    protect: SelectorT,
    #    file_writer: StringFileWriter,
    #    frame_scale: float,
    #    local_configs: dict[SelectorT, dict[str, str]],
    #    global_attrs: dict[str, str]
    #) -> None:

    #    #def get_content_by_body(
    #    #    body: str,
    #    #    is_labelled: bool
    #    #) -> str:
    #    #    prefix, suffix = tuple(
    #    #        self.get_command_string(
    #    #            global_attrs,
    #    #            edge_flag=edge_flag,
    #    #            label=0 if is_labelled else None
    #    #        )
    #    #        for edge_flag in (EdgeFlag.START, EdgeFlag.STOP)
    #    #    )
    #    #    return "".join((prefix, body, suffix))

    #    super().__init__(
    #        string=string,
    #        isolate=isolate,
    #        protect=protect,
    #        global_attrs=global_attrs,
    #        local_attrs=local_configs,
    #        #configured_items_iterator=(
    #        #    (span, local_config)
    #        #    for selector, local_config in local_configs.items()
    #        #    for span in self.iter_spans_by_selector(selector, string)
    #        #),
    #        #get_content_by_body=get_content_by_body,
    #        file_writer=file_writer,
    #        frame_scale=frame_scale
    #    )

    @classmethod
    def escape_markup_char(
        cls,
        substr: str
    ) -> str:
        return cls._MARKUP_ENTITY_DICT.get(substr, substr)

    @classmethod
    def unescape_markup_char(
        cls,
        substr: str
    ) -> str:
        return cls._MARKUP_ENTITY_REVERSED_DICT.get(substr, substr)

    @classmethod
    def iter_command_matches(
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
    def get_command_flag(
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
    def replace_for_content(
        cls,
        match_obj: re.Match[str]
    ) -> str:
        if match_obj.group("tag"):
            return ""
        if match_obj.group("char"):
            return cls.escape_markup_char(match_obj.group("char"))
        return match_obj.group()

    @classmethod
    def replace_for_matching(
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
            return cls.unescape_markup_char(match_obj.group("entity"))
        return match_obj.group()

    @classmethod
    def get_attrs_from_command_pair(
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
    def get_command_string(
        cls,
        attrs: dict[str, str],
        edge_flag: EdgeFlag,
        label: int | None
    ) -> str:
        if edge_flag == EdgeFlag.STOP:
            return "</span>"

        if label is not None:
            converted_attrs = {"foreground": f"#{label:06x}"}
            for key, val in attrs.items():
                if key in (
                    "background", "bgcolor",
                    "underline_color", "overline_color", "strikethrough_color"
                ):
                    converted_attrs[key] = "black"
                elif key not in ("foreground", "fgcolor", "color"):
                    converted_attrs[key] = val
        else:
            converted_attrs = attrs.copy()
        attrs_str = " ".join([
            f"{key}='{val}'"
            for key, val in converted_attrs.items()
        ])
        return f"<span {attrs_str}>"


class TextParser(MarkupTextParser):
    __slots__ = ()

    @classmethod
    def iter_command_matches(
        cls,
        string: str
    ) -> Iterator[re.Match[str]]:
        pattern = re.compile(r"""[<>&"']""")
        yield from pattern.finditer(string)

    @classmethod
    def get_command_flag(
        cls,
        match_obj: re.Match[str]
    ) -> CommandFlag:
        return CommandFlag.OTHER

    @classmethod
    def replace_for_content(
        cls,
        match_obj: re.Match[str]
    ) -> str:
        return cls.escape_markup_char(match_obj.group())

    @classmethod
    def replace_for_matching(
        cls,
        match_obj: re.Match[str]
    ) -> str:
        return match_obj.group()


class Text(StringMobject):
    __slots__ = ()

    _TEXT_SCALE_FACTOR: ClassVar[float] = 0.0076  # TODO

    def __init__(
        self,
        string: str,
        *,
        isolate: SelectorT = (),
        protect: SelectorT = (),
        local_configs: dict[SelectorT, dict[str, str]] = ...,
        justify: bool = ...,
        indent: float = ...,
        alignment: str = ...,
        line_width: float | None = ...,
        font_size: float = ...,
        font: str = ...,
        slant: str = ...,
        weight: str = ...,
        base_color: ColorT = ...,
        line_spacing_height: float = ...,
        global_config: dict[str, str] = ...,
        markup: bool = False
    ) -> None:
        if markup:
            PangoUtils.validate_markup_string(string)
        if local_configs is ...:
            local_configs = {}

        config = ConfigSingleton().text
        if justify is ...:
            justify = config.justify
        if indent is ...:
            indent = config.indent
        if alignment is ...:
            alignment = config.alignment
        if line_width is ...:
            line_width = config.line_width
        if font_size is ...:
            font_size = config.font_size
        if font is ...:
            font = config.font
        if slant is ...:
            slant = config.slant
        if weight is ...:
            weight = config.weight
        if base_color is ...:
            base_color = config.base_color
        if line_spacing_height is ...:
            line_spacing_height = config.line_spacing_height
        if global_config is ...:
            global_config = config.global_config

        global_attrs = {
            "font_size": str(round(font_size * 1024.0)),
            "font_family": font,
            "font_style": slant,
            "font_weight": weight,
            "foreground": ColorUtils.color_to_hex(base_color),
            "line_height": str(1.0 + line_spacing_height)
        }
        global_attrs.update(global_config)

        file_writer = MarkupTextFileWriter(
            justify=justify,
            indent=indent,
            alignment=PangoAlignment[alignment],
            line_width=line_width
        )
        parser_class = MarkupTextParser if markup else TextParser
        parser = parser_class(
            string=string,
            isolate=isolate,
            protect=protect,
            global_attrs=global_attrs,
            local_attrs=local_configs,
            file_writer=file_writer,
            frame_scale=self._TEXT_SCALE_FACTOR
        )
        super().__init__(
            string=string,
            parser=parser
        )

    #@classmethod
    #def _get_global_attrs(
    #    cls,
    #    font_size: float,
    #    font: str,
    #    slant: str,
    #    weight: str,
    #    base_color: ColorT,
    #    line_spacing_height: float,
    #    global_config: dict[str, str]
    #) -> dict[str, str]:
    #    global_attrs = {
    #        "font_size": str(round(font_size * 1024.0)),
    #        "font_family": font,
    #        "font_style": slant,
    #        "font_weight": weight,
    #        "foreground": ColorUtils.color_to_hex(base_color),
    #        "line_height": str(1.0 + line_spacing_height)
    #    }
    #    global_attrs.update(global_config)
    #    return global_attrs


class Code(Text):
    __slots__ = ()

    def __init__(
        self,
        code: str,
        *,
        language: str = ...,
        code_style: str = ...,
        **kwargs
    ) -> None:
        config = ConfigSingleton().text
        if language is ...:
            language = config.language
        if code_style is ...:
            code_style = config.code_style

        lexer = pygments.lexers.get_lexer_by_name(language)
        formatter = pygments.formatters.PangoMarkupFormatter(
            style=code_style
        )
        markup_string = pygments.highlight(code, lexer, formatter)
        markup_string = re.sub(r"</?tt>", "", markup_string)
        return super().__init__(
            string=markup_string,
            markup=True,
            **kwargs
        )
