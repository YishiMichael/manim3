import json
import os
import pathlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import (
    ClassVar,
    Iterable,
    Iterator,
    TypeVar,
    TypedDict
)

import manimpango

from ...constants.custom_typing import (
    AlignmentT,
    ColorT,
    SelectorT
)
from ...toplevel.toplevel import Toplevel
from ...utils.color_utils import ColorUtils
from .string_mobject import (
    CommandFlag,
    EdgeFlag,
    Span,
    StringMobject,
    StringMobjectIO,
    StringMobjectInputData
)


_PangoBaseInputDataT = TypeVar("_PangoBaseInputDataT", bound="PangoBaseInputData")


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


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class PangoBaseInputData(StringMobjectInputData):
    color: ColorT
    font_size: float
    alignment: AlignmentT
    font: str
    justify: bool
    indent: float
    line_width: float
    global_config: dict[str, str]
    local_colors: dict[Span, ColorT]
    local_configs: dict[Span, dict[str, str]]


class PangoBaseIO(StringMobjectIO[_PangoBaseInputDataT]):
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
    @property
    def _temp_extensions(cls) -> list[str]:
        return [".svg"]

    @classmethod
    def _get_global_attrs(
        cls,
        input_data: _PangoBaseInputDataT,
        temp_path: pathlib.Path
    ) -> dict[str, str]:
        global_attrs = {
            "foreground": ColorUtils.color_to_hex(input_data.color),
            "font_size": str(round(input_data.font_size * 1024.0)),
            "font_family": input_data.font
        }
        global_attrs.update(input_data.global_config)
        return global_attrs

    @classmethod
    def _get_local_attrs(
        cls,
        input_data: _PangoBaseInputDataT,
        temp_path: pathlib.Path
    ) -> dict[Span, dict[str, str]]:
        local_attrs = {
            span: {
                "foreground": ColorUtils.color_to_hex(local_color)
            }
            for span, local_color in input_data.local_colors.items()
        }
        for span, local_config in input_data.local_configs.items():
            local_attrs.setdefault(span, {}).update(local_config)
        return local_attrs

    @classmethod
    def _create_svg(
        cls,
        content: str,
        input_data: _PangoBaseInputDataT,
        svg_path: pathlib.Path
    ) -> None:
        justify = input_data.justify
        indent = input_data.indent
        alignment = input_data.alignment
        line_width = input_data.line_width

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

    @classmethod
    def _get_svg_frame_scale(
        cls,
        input_data: _PangoBaseInputDataT
    ) -> float:
        return 0.01147

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


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class MarkupInputData(PangoBaseInputData):
    pass


class MarkupIO(PangoBaseIO[MarkupInputData]):
    __slots__ = ()

    @classmethod
    @property
    def _dir_name(cls) -> str:
        return "markup"

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


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TextInputData(PangoBaseInputData):
    pass


class TextIO(PangoBaseIO[TextInputData]):
    __slots__ = ()

    @classmethod
    @property
    def _dir_name(cls) -> str:
        return "text"


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class MarkupWritingConfig(StringMobjectWritingConfig):
#    justify: bool
#    indent: float
#    alignment: AlignmentT
#    line_width: float


#class MarkupIO(StringMobjectIO[MarkupWritingConfig]):
#    __slots__ = ()

#    _dir_name: ClassVar[str] = "markup"
#    #_parser_cls: ClassVar[type[StringParser]] = MarkupParser
#    _TEXT_SCALE_FACTOR: ClassVar[float] = 0.01147

#    @classmethod
#    def _get_shape_mobjects(
#        cls,
#        content: str,
#        writing_config: MarkupWritingConfig,
#        temp_path: pathlib.Path
#    ) -> list[ShapeMobject]:
#        justify = writing_config.justify
#        indent = writing_config.indent
#        alignment = writing_config.alignment
#        line_width = writing_config.line_width

#        svg_path = temp_path.with_suffix(".svg")

#        try:
#            PangoUtils.validate_markup_string(content)
#            PangoUtils.create_markup_svg(
#                markup_str=content,
#                svg_path=svg_path,
#                justify=justify,
#                indent=indent,
#                alignment=alignment,
#                pango_width=(
#                    -1 if line_width < 0.0
#                    else line_width * Toplevel.config.pixel_per_unit
#                )
#            )

#            shape_mobjects = list(SVGMobjectIO.iter_shape_mobject_from_svg(
#                svg_path=svg_path,
#                frame_scale=cls._TEXT_SCALE_FACTOR
#            ))

#        finally:
#            temp_path.with_suffix(".svg").unlink(missing_ok=True)

#        return shape_mobjects


class PangoBaseStringMobject(StringMobject):
    __slots__ = ()

    #_parser_cls: ClassVar[type[StringParser]] = MarkupParser
    #_dir_name: ClassVar[str] = "_markup"
    #_io_cls: ClassVar[type[StringMobjectIO]] = MarkupIO

    def __init__(
        self,
        string: str,
        *,
        isolate: Iterable[SelectorT] = (),
        protect: Iterable[SelectorT] = (),
        color: ColorT | None = None,
        font_size: float | None = None,
        alignment: AlignmentT | None = None,
        font: str | None = None,
        justify: bool | None = None,
        indent: float | None = None,
        line_width: float | None = None,
        global_config: dict[str, str] | None = None,
        local_colors: dict[SelectorT, ColorT] | None = None,
        local_configs: dict[SelectorT, dict[str, str]] | None = None,
        **kwargs
    ) -> None:
        config = Toplevel.config
        if color is None:
            color = config.text_color
        if font_size is None:
            font_size = config.text_font_size
        if alignment is None:
            alignment = config.text_alignment
        if font is None:
            font = config.text_font
        if justify is None:
            justify = config.text_justify
        if indent is None:
            indent = config.text_indent
        if line_width is None:
            line_width = config.text_line_width
        if global_config is None:
            global_config = {}
        if local_colors is None:
            local_colors = {}
        if local_configs is None:
            local_configs = {}

        #global_attrs = {
        #    "foreground": ColorUtils.color_to_hex(color),
        #    "font_size": str(round(font_size * 1024.0)),
        #    "font_family": font
        #}
        #global_attrs.update(global_config)

        #local_attrs = {
        #    selector: {
        #        "foreground": ColorUtils.color_to_hex(local_color)
        #    }
        #    for selector, local_color in local_colors.items()
        #}
        #for selector, local_config in local_configs.items():
        #    local_attrs.setdefault(selector, {}).update(local_config)

        #super().__init__(
        #    string=string,
        #    isolate=isolate,
        #    protect=protect,
        #    global_attrs=global_attrs,
        #    local_attrs=local_attrs,
        #    writing_config=MarkupWritingConfig(
        #        justify=justify,
        #        indent=indent,
        #        alignment=alignment,
        #        line_width=line_width
        #    )
        #)
        cls = type(self)
        super().__init__(
            string=string,
            isolate=cls._get_spans_by_selectors(isolate, string),
            protect=cls._get_spans_by_selectors(protect, string),
            color=color,
            font_size=font_size,
            alignment=alignment,
            font=font,
            justify=justify,
            indent=indent,
            line_width=line_width,
            global_config=global_config,
            local_colors={
                span: local_color
                for selector, local_color in local_colors.items()
                for span in self._iter_spans_by_selector(selector, string)
            },
            local_configs={
                span: local_config
                for selector, local_config in local_configs.items()
                for span in self._iter_spans_by_selector(selector, string)
            },
            **kwargs
        )

class Markup(PangoBaseStringMobject):
    __slots__ = ()

    @classmethod
    @property
    def _io_cls(cls) -> type[StringMobjectIO[MarkupInputData]]:
        return MarkupIO

    @classmethod
    @property
    def _input_data_cls(cls) -> type[MarkupInputData]:
        return MarkupInputData

    #@classmethod
    #def _create_svg_file(
    #    cls,
    #    content: str,
    #    svg_path: pathlib.Path,
    #    justify: bool,
    #    indent: float,
    #    alignment: AlignmentT,
    #    line_width: float
    #) -> None:
    #    PangoUtils.validate_markup_string(content)
    #    PangoUtils.create_markup_svg(
    #        markup_str=content,
    #        svg_path=svg_path,
    #        justify=justify,
    #        indent=indent,
    #        alignment=alignment,
    #        pango_width=(
    #            -1 if line_width < 0.0
    #            else line_width * Toplevel.config.pixel_per_unit
    #        )
    #    )


class Text(PangoBaseStringMobject):
    __slots__ = ()

    #def __init__(
    #    self,
    #    string: str,
    #    *,
    #    isolate: Iterable[SelectorT] = (),
    #    protect: Iterable[SelectorT] = (),
    #    color: ColorT | None = None,
    #    font_size: float | None = None,
    #    alignment: AlignmentT | None = None,
    #    font: str | None = None,
    #    justify: bool | None = None,
    #    indent: float | None = None,
    #    line_width: float | None = None,
    #    global_config: dict[str, str] | None = None,
    #    local_colors: dict[SelectorT, ColorT] | None = None,
    #    local_configs: dict[SelectorT, dict[str, str]] | None = None
    #) -> None:
    #    config = Toplevel.config
    #    if color is None:
    #        color = config.text_color
    #    if font_size is None:
    #        font_size = config.text_font_size
    #    if alignment is None:
    #        alignment = config.text_alignment
    #    if font is None:
    #        font = config.text_font
    #    if justify is None:
    #        justify = config.text_justify
    #    if indent is None:
    #        indent = config.text_indent
    #    if line_width is None:
    #        line_width = config.text_line_width
    #    if global_config is None:
    #        global_config = {}
    #    if local_colors is None:
    #        local_colors = {}
    #    if local_configs is None:
    #        local_configs = {}

    #    cls = type(self)
    #    super().__init__(TextInputData(
    #        string=string,
    #        isolate=cls._get_spans_by_selectors(isolate, string),
    #        protect=cls._get_spans_by_selectors(protect, string),
    #        color=color,
    #        font_size=font_size,
    #        alignment=alignment,
    #        font=font,
    #        justify=justify,
    #        indent=indent,
    #        line_width=line_width,
    #        global_config=global_config,
    #        local_colors={
    #            span: local_color
    #            for selector, local_color in local_colors.items()
    #            for span in self._iter_spans_by_selector(selector, string)
    #        },
    #        local_configs={
    #            span: local_config
    #            for selector, local_config in local_configs.items()
    #            for span in self._iter_spans_by_selector(selector, string)
    #        }
    #    ))

    @classmethod
    @property
    def _io_cls(cls) -> type[StringMobjectIO[TextInputData]]:
        return TextIO

    @classmethod
    @property
    def _input_data_cls(cls) -> type[TextInputData]:
        return TextInputData


# Save at `~\Sublime Text\Packages\export_highlight\export_highlight.py`
"""
import json

import sublime
import sublime_plugin


class ExportHighlightCommand(sublime_plugin.ApplicationCommand):
    def run(self):
        window = sublime.active_window()
        view = window.active_view()
        if view.is_loading():
            return

        full_region = sublime.Region(0, view.size())
        tokens = [
            {
                "begin": region.begin(),
                "end": region.end(),
                "style": view.style_for_scope(scope)
            }
            for region, scope in view.extract_tokens_with_scopes(full_region)
        ]
        stem, _ = view.file_name().rsplit(".", maxsplit=1)
        with open(stem + ".json", "w") as output_file:
            json.dump(tokens, output_file)

        view.close()
"""


# https://www.sublimetext.com/docs/api_reference.html#sublime.View.style_for_scope
class _HighlightStyle(TypedDict, total=False):
    foreground: str
    background: str
    bold: bool
    italic: bool
    #glow
    #underline
    #stippled_underline
    #squiggly_underline
    source_line: int
    source_column: int
    source_file: str


class _Token(TypedDict):
    begin: int
    end: int
    style: _HighlightStyle


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class CodeInputData(PangoBaseInputData):
    language_extension: str


class CodeIO(PangoBaseIO[CodeInputData]):
    __slots__ = ()

    @classmethod
    @property
    def _dir_name(cls) -> str:
        return "code"

    @classmethod
    def _get_local_attrs(
        cls,
        input_data: CodeInputData,
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

        language_extension = input_data.language_extension
        try:
            code_path = temp_path.with_suffix(language_extension)
            with open(code_path, "w", encoding="utf-8") as code_file:
                code_file.write(input_data.string)

            # First open the file, then launch the command.
            # We separate these two steps as file loading is asynchronous,
            # and operations on `view` has no effect while loading.
            os.system(" ".join((
                "subl",
                "--background",   # Don't activate the application.
                f"\"{code_path}\""
            )))
            os.system(" ".join((
                "subl",
                "--background",
                "--command", "export_highlight"
            )))

            with open(temp_path.with_suffix(".json")) as input_file:
                local_attrs = dict(
                    local_config_from_token(token)
                    for token in json.load(input_file)
                )
        finally:
            for extension in (language_extension, ".json"):
                temp_path.with_suffix(extension).unlink(missing_ok=True)

        for span, local_config in super()._get_local_attrs(input_data, temp_path).items():
            local_attrs.setdefault(span, {}).update(local_config)
        return local_attrs


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class CodeWritingConfig(MarkupWritingConfig):
#    language_extension: str


#class CodeIO(MarkupIO):
#    __slots__ = ()

#    _dir_name: ClassVar[str] = "code"

#    @classmethod
#    def _get_shape_mobjects(
#        cls,
#        content: str,
#        writing_config: CodeWritingConfig,
#        temp_path: pathlib.Path
#    ) -> list[ShapeMobject]:
#        justify = writing_config.justify
#        indent = writing_config.indent
#        alignment = writing_config.alignment
#        line_width = writing_config.line_width

#        svg_path = temp_path.with_suffix(".svg")

#        try:
#            PangoUtils.validate_markup_string(content)
#            PangoUtils.create_markup_svg(
#                markup_str=content,
#                svg_path=svg_path,
#                justify=justify,
#                indent=indent,
#                alignment=alignment,
#                pango_width=(
#                    -1 if line_width < 0.0
#                    else line_width * Toplevel.config.pixel_per_unit
#                )
#            )

#            shape_mobjects = list(SVGMobjectIO.iter_shape_mobject_from_svg(
#                svg_path=svg_path,
#                frame_scale=cls._TEXT_SCALE_FACTOR
#            ))

#        finally:
#            temp_path.with_suffix(".svg").unlink(missing_ok=True)

#        return shape_mobjects


class Code(PangoBaseStringMobject):
    __slots__ = ()

    def __init__(
        self,
        string: str,
        *,
        font: str | None = None,
        language_extension: str | None = None,
        **kwargs
    ) -> None:
        config = Toplevel.config
        if font is None:
            font = config.code_font
        if language_extension is None:
            language_extension = config.code_language_extension

        super().__init__(
            string=string,
            font=font,
            language_extension=language_extension,
            **kwargs
        )

    @classmethod
    @property
    def _io_cls(cls) -> type[StringMobjectIO[CodeInputData]]:
        return CodeIO

    @classmethod
    @property
    def _input_data_cls(cls) -> type[CodeInputData]:
        return CodeInputData

    #def __init__(
    #    self,
    #    code: str,
    #    *,
    #    language_extension: str | None = None,
    #    isolate: Iterable[SelectorT] = (),
    #    protect: Iterable[SelectorT] = (),
    #    color: ColorT | None = None,
    #    font_size: float | None = None,
    #    alignment: AlignmentT | None = None,
    #    font: str | None = None,
    #    justify: bool | None = None,
    #    indent: float | None = None,
    #    line_width: float | None = None,
    #    global_config: dict[str, str] | None = None,
    #    local_colors: dict[SelectorT, ColorT] | None = None,
    #    local_configs: dict[SelectorT, dict[str, str]] | None = None
    #) -> None:
    #    config = Toplevel.config
    #    if font is None:
    #        font = config.code_font
    #    if language_extension is None:
    #        language_extension = config.code_language_extension
    #    if local_configs is None:
    #        local_configs = {}

    #    code_path = StringFileWriter.get_hash_path(
    #        hash_content=str((code, language_extension)),
    #        dir_name="_code",
    #        suffix=language_extension
    #    )
    #    #code_path = self.get_code_path(code, suffix)
    #    json_path = code_path.with_suffix(".json")
    #    if not json_path.exists():
    #        with open(code_path, "w", encoding="utf-8") as code_file:
    #            code_file.write(code)

    #        # First open the file, then launch the command.
    #        # We separate these two steps as file loading is asynchronous,
    #        # and operations on `view` has no effect while loading.
    #        os.system(" ".join((
    #            "subl",
    #            "--background",   # Don't activate the application.
    #            f"\"{code_path}\""
    #        )))
    #        os.system(" ".join((
    #            "subl",
    #            "--background",
    #            "--command", "export_highlight"
    #        )))

    #    def local_config_from_token(
    #        token: _Token
    #    ) -> tuple[tuple[int, int], dict[str, str]]:
    #        style = token["style"]
    #        config: dict[str, str] = {}
    #        if (foreground := style.get("foreground")):
    #            config["fgcolor"] = foreground
    #        if (background := style.get("background")):
    #            config["bgcolor"] = background
    #        if style.get("bold"):
    #            config["font_weight"] = "bold"
    #        if style.get("italic"):
    #            config["font_style"] = "italic"
    #        return (token["begin"], token["end"]), config

    #    with open(json_path) as input_file:
    #        token_configs: dict[SelectorT, dict[str, str]] = dict(
    #            local_config_from_token(token)
    #            for token in json.load(input_file)
    #        )

    #    for selector, local_config in local_configs.items():
    #        token_configs.setdefault(selector, {}).update(local_config)
    #    super().__init__(
    #        string=code,
    #        font=font,
    #        local_configs=token_configs,
    #        **kwargs
    #    )

    #@classmethod
    #def get_code_path(
    #    cls,
    #    code: str,
    #    suffix: str
    #) -> pathlib.Path:
    #    hash_content = str((code, suffix))
    #    # Truncating at 16 bytes for cleanliness.
    #    hex_string = hashlib.sha256(hash_content.encode()).hexdigest()[:16]
    #    code_dir = PathUtils.get_output_subdir("_code")
    #    return code_dir.joinpath(f"{hex_string}{suffix}")
