__all__ = [
    "Text",
    "Code"
]


from enum import Enum
import pathlib
import re
from typing import (
    ClassVar,
    Iterator
)
import warnings

import manimpango
import pygments
import pygments.formatters
import pygments.lexers

from ..custom_typing import (
    ColorT,
    SelectorT
)
from ..mobjects.string_mobject import (
    CommandFlag,
    EdgeFlag,
    StringFileWriter,
    StringMobject,
    StringParser
)
from ..rendering.config import ConfigSingleton
from ..utils.color import ColorUtils


# Ported from `manimpango/enums.pyx`.
class PangoAlignment(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2


class PangoUtils:
    # Ensure the canvas is large enough to hold all glyphs.
    _DEFAULT_CANVAS_WIDTH: ClassVar[int] = 16384
    _DEFAULT_CANVAS_HEIGHT: ClassVar[int] = 16384

    def __new__(cls):
        raise TypeError

    @classmethod
    @property
    def pango_version_str(cls) -> str:
        return manimpango.pango_version()

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
            font="",                     # Already handled
            slant="NORMAL",              # Already handled
            weight="NORMAL",             # Already handled
            size=1,                      # Already handled
            _=0,                         # Empty parameter
            disable_liga=False,
            file_name=str(svg_path),
            START_X=0,
            START_Y=0,
            width=cls._DEFAULT_CANVAS_WIDTH,
            height=cls._DEFAULT_CANVAS_HEIGHT,
            justify=justify,
            indent=indent,
            line_spacing=None,           # Already handled
            alignment=alignment,
            pango_width=pango_width
        )


class MarkupTextFileWriter(StringFileWriter):
    __slots__ = (
        "_justify",
        "_indent",
        "_alignment",
        "_line_width"
    )

    def __init__(
        self,
        justify: bool,
        indent: float,
        alignment: PangoAlignment,
        line_width: float | None
    ) -> None:
        super().__init__()
        self._justify: bool = justify
        self._indent: float = indent
        self._alignment: PangoAlignment = alignment
        self._line_width: float | None = line_width

    def get_svg_path(
        self,
        content: str
    ) -> pathlib.Path:
        hash_content = str((
            content,
            self._justify,
            self._indent,
            self._alignment,
            self._line_width
        ))
        return ConfigSingleton().path.text_dir.joinpath(f"{self.hash_string(hash_content)}.svg")

    def create_svg_file(
        self,
        content: str,
        svg_path: pathlib.Path
    ) -> None:
        PangoUtils.validate_markup_string(content)
        PangoUtils.create_markup_svg(
            markup_str=content,
            svg_path=svg_path,
            justify=self._justify,
            indent=self._indent,
            alignment=self._alignment,
            pango_width=(
                -1 if (line_width := self._line_width) is None
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

    def __init__(
        self,
        string: str,
        isolate: SelectorT,
        protect: SelectorT,
        file_writer: StringFileWriter,
        frame_scale: float,
        local_configs: dict[SelectorT, dict[str, str]],
        global_attrs: dict[str, str]
    ) -> None:

        def get_content_by_body(
            body: str,
            is_labelled: bool
        ) -> str:
            prefix, suffix = tuple(
                self._get_command_string(
                    global_attrs,
                    edge_flag=edge_flag,
                    label=0 if is_labelled else None
                )
                for edge_flag in (EdgeFlag.START, EdgeFlag.STOP)
            )
            return "".join((prefix, body, suffix))

        super().__init__(
            string=string,
            isolate=isolate,
            protect=protect,
            configured_items_iterator=(
                (span, local_config)
                for selector, local_config in local_configs.items()
                for span in self._iter_spans_by_selector(selector, string)
            ),
            get_content_by_body=get_content_by_body,
            file_writer=file_writer,
            frame_scale=frame_scale
        )

    @classmethod
    def _escape_markup_char(
        cls,
        substr: str
    ) -> str:
        return cls._MARKUP_ENTITY_DICT.get(substr, substr)

    @classmethod
    def _unescape_markup_char(
        cls,
        substr: str
    ) -> str:
        return cls._MARKUP_ENTITY_REVERSED_DICT.get(substr, substr)

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
            return cls._escape_markup_char(match_obj.group("char"))
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
            return cls._unescape_markup_char(match_obj.group("entity"))
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
        return cls._escape_markup_char(match_obj.group())

    @classmethod
    def _replace_for_matching(
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
        local_configs: dict[SelectorT, dict[str, str]] | None = None,
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
        if local_configs is None:
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

        global_attrs = self._get_global_attrs(
            font_size=font_size,
            font=font,
            slant=slant,
            weight=weight,
            base_color=base_color,
            line_spacing_height=line_spacing_height,
            global_config=global_config
        )
        parser_class = MarkupTextParser if markup else TextParser
        parser = parser_class(
            string=string,
            isolate=isolate,
            protect=protect,
            file_writer=MarkupTextFileWriter(
                justify=justify,
                indent=indent,
                alignment=PangoAlignment[alignment],
                line_width=line_width
            ),
            frame_scale=self._TEXT_SCALE_FACTOR,
            local_configs=local_configs,
            global_attrs=global_attrs
        )

        super().__init__(
            string=string,
            parser=parser
        )

    @classmethod
    def _get_global_attrs(
        cls,
        font_size: float,
        font: str,
        slant: str,
        weight: str,
        base_color: ColorT,
        line_spacing_height: float,
        global_config: dict[str, str]
    ) -> dict[str, str]:
        global_attrs = {
            "font_size": str(round(font_size * 1024.0)),
            "font_family": font,
            "font_style": slant,
            "font_weight": weight,
            "foreground": ColorUtils.color_to_hex(base_color)
        }
        # `line_height` attribute is supported since Pango 1.50.
        pango_version_str = PangoUtils.pango_version_str
        if tuple(map(int, pango_version_str.split("."))) < (1, 50):
            warnings.warn(
                f"Pango version {pango_version_str} found (< 1.50), " +
                "unable to set `line_height` attribute"
            )
        else:
            global_attrs["line_height"] = str(1.0 + line_spacing_height)

        global_attrs.update(global_config)
        return global_attrs


class Code(Text):
    __slots__ = ()

    def __init__(
        self,
        code: str,
        *,
        language: str | None = None,
        code_style: str | None = None,
        **kwargs
    ) -> None:
        config = ConfigSingleton().text
        if language is None:
            language = config.language
        if code_style is None:
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
