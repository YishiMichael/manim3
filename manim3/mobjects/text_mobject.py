__all__ = [
    "MarkupText",
    "Text",
    "Code"
]


import hashlib
import os
import re
from typing import (
    ClassVar,
    Generator
)
import warnings

from colour import Color
import manimpango
import pygments
import pygments.formatters
import pygments.lexers

from ..custom_typing import (
    ColorType,
    Real,
    Selector
)
from ..mobjects.string_mobject import (
    CommandFlag,
    EdgeFlag,
    StringMobject
)
from ..rendering.config import ConfigSingleton
from ..utils.color import ColorUtils


def hash_string(
    string: str
) -> str:  # TODO: redundant with tex_mobject.py
    # Truncating at 16 bytes for cleanliness
    hasher = hashlib.sha256(string.encode())
    return hasher.hexdigest()[:16]


TEXT_MOB_SCALE_FACTOR = 0.0076
DEFAULT_LINE_SPACING_SCALE = 0.6
# Ensure the canvas is large enough to hold all glyphs.
DEFAULT_CANVAS_WIDTH = 16384
DEFAULT_CANVAS_HEIGHT = 16384


# TODO: use Enum
# Temporary handler
class _Alignment:
    VAL_DICT = {
        "LEFT": 0,
        "CENTER": 1,
        "RIGHT": 2
    }

    def __init__(
        self,
        s: str
    ):
        self.value = _Alignment.VAL_DICT[s.upper()]


class MarkupText(StringMobject):
    __slots__ = ()

    # See https://docs.gtk.org/Pango/pango_markup.html
    MARKUP_TAGS: ClassVar[dict[str, dict[str, str]]] = {
        "b": {"font_weight": "bold"},
        "big": {"font_size": "larger"},
        "i": {"font_style": "italic"},
        "s": {"strikethrough": "true"},
        "sub": {"baseline_shift": "subscript", "font_scale": "subscript"},
        "sup": {"baseline_shift": "superscript", "font_scale": "superscript"},
        "small": {"font_size": "smaller"},
        "tt": {"font_family": "monospace"},
        "u": {"underline": "single"},
    }
    MARKUP_ENTITY_DICT: ClassVar[dict[str, str]] = {
        "<": "&lt;",
        ">": "&gt;",
        "&": "&amp;",
        "\"": "&quot;",
        "'": "&apos;"
    }

    def __init__(
        self,
        string: str,
        *,
        font_size: Real = 48,
        line_spacing_height: Real | None = None,
        justify: bool = False,
        indent: Real = 0.0,
        alignment: str | None = None,
        line_width: Real | None = None,
        font: str | None = None,
        slant: str = "NORMAL",
        weight: str = "NORMAL",
        base_color: ColorType = Color("white"),
        global_config: dict[str, str] | None = None,
        local_configs: dict[Selector, dict[str, str]] | None = None,
        disable_ligatures: bool = True,
        isolate: Selector = re.compile(r"\w+", flags=re.UNICODE),
        protect: Selector = (),
        width: Real | None = None,
        height: Real | None = None
    ):
        if alignment is None:
            alignment = "LEFT"  # TODO
        if font is None:
            font = "Consolas"  # TODO
        if global_config is None:
            global_config = {}
        if local_configs is None:
            local_configs = {}

        if not isinstance(self, Text):
            self._validate_markup_string(string)
        #if not self.font:
        #    self.font = get_customization()["style"]["font"]
        #if not self.alignment:
        #    self.alignment = get_customization()["style"]["text_alignment"]

        def get_content_prefix_and_suffix(
            is_labelled: bool
        ) -> tuple[str, str]:
            global_attr_dict = {
                "foreground": ColorUtils.color_to_hex(base_color),
                "font_family": font,
                "font_style": slant,
                "font_weight": weight,
                "font_size": str(round(font_size * 1024.0)),
            }
            # `line_height` attribute is supported since Pango 1.50.
            pango_version = manimpango.pango_version()
            if tuple(map(int, pango_version.split("."))) < (1, 50):
                if line_spacing_height is not None:
                    warnings.warn(
                        f"Pango version {pango_version} found (< 1.50), unable to set `line_height` attribute"
                    )
            else:
                line_spacing_scale = line_spacing_height or DEFAULT_LINE_SPACING_SCALE
                global_attr_dict["line_height"] = str(
                    ((line_spacing_scale) + 1) * 0.6
                )
            if disable_ligatures:
                global_attr_dict["font_features"] = "liga=0,dlig=0,clig=0,hlig=0"

            global_attr_dict.update(global_config)
            return tuple(
                self._get_command_string(
                    global_attr_dict,
                    edge_flag=edge_flag,
                    label=0 if is_labelled else None
                )
                for edge_flag in (EdgeFlag.START, EdgeFlag.STOP)
            )

        def get_svg_path(
            content: str
        ) -> str:
            hash_content = str((
                content,
                justify,
                indent,
                alignment,
                line_width
            ))
            svg_file = os.path.join(
                ConfigSingleton().text_dir, f"{hash_string(hash_content)}.svg"
            )
            if not os.path.exists(svg_file):
                markup_to_svg(content, svg_file)
            return svg_file

        def markup_to_svg(
            markup_str: str,
            file_name: str
        ) -> str:
            self._validate_markup_string(markup_str)

            # `manimpango` is under construction,
            # so the following code is intended to suit its interface
            if line_width is None:
                pango_width = -1
            else:
                pango_width = line_width * ConfigSingleton().pixel_per_unit

            return manimpango.MarkupUtils.text2svg(
                text=markup_str,
                font="",                     # Already handled
                slant="NORMAL",              # Already handled
                weight="NORMAL",             # Already handled
                size=1,                      # Already handled
                _=0,                         # Empty parameter
                disable_liga=False,
                file_name=file_name,
                START_X=0,
                START_Y=0,
                width=DEFAULT_CANVAS_WIDTH,
                height=DEFAULT_CANVAS_HEIGHT,
                justify=justify,
                indent=indent,
                line_spacing=None,           # Already handled
                alignment=_Alignment(alignment),
                pango_width=pango_width
            )

        super().__init__(
            string=string,
            isolate=isolate,
            protect=protect,
            configured_items_generator=(
                (span, local_config)
                for selector, local_config in local_configs.items()
                for span in cls._iter_spans_by_selector(selector, string)
            ),
            get_content_prefix_and_suffix=get_content_prefix_and_suffix,
            get_svg_path=get_svg_path,
            width=width,
            height=height,
            frame_scale=TEXT_MOB_SCALE_FACTOR
        )

    @classmethod
    def _validate_markup_string(
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

    # Toolkits

    @classmethod
    def _escape_markup_char(
        cls,
        substr: str
    ) -> str:
        return cls.MARKUP_ENTITY_DICT.get(substr, substr)

    @classmethod
    def _unescape_markup_char(
        cls,
        substr: str
    ) -> str:
        return {
            v: k
            for k, v in cls.MARKUP_ENTITY_DICT.items()
        }.get(substr, substr)

    # Parsing

    @classmethod
    def _iter_command_matches(
        cls,
        string: str
    ) -> Generator[re.Match[str], None, None]:
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
    def _get_attr_dict_from_command_pair(
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
        return cls.MARKUP_TAGS.get(tag_name, {})

    @classmethod
    def _get_command_string(
        cls,
        attr_dict: dict[str, str],
        edge_flag: EdgeFlag,
        label: int | None
    ) -> str:
        if edge_flag == EdgeFlag.STOP:
            return "</span>"

        if label is not None:
            converted_attr_dict = {"foreground": f"#{label:06x}"}
            for key, val in attr_dict.items():
                if key in (
                    "background", "bgcolor",
                    "underline_color", "overline_color", "strikethrough_color"
                ):
                    converted_attr_dict[key] = "black"
                elif key not in ("foreground", "fgcolor", "color"):
                    converted_attr_dict[key] = val
        else:
            converted_attr_dict = attr_dict.copy()
        attrs_str = " ".join([
            f"{key}='{val}'"
            for key, val in converted_attr_dict.items()
        ])
        return f"<span {attrs_str}>"


class Text(MarkupText):
    #CONFIG = {
    #    # For backward compatibility
    #    "isolate": (re.compile(r"\w+", re.U), re.compile(r"\S+", re.U)),
    #}

    @classmethod
    def _iter_command_matches(
        cls,
        string: str
    ) -> Generator[re.Match[str], None, None]:
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


class Code(MarkupText):
    __slots__ = ()

    def __init__(
        self,
        code: str,
        *,
        language: str = "python",
        # Visit https://pygments.org/demo/ to have a preview of more styles.
        code_style: str = "monokai",
        font_size: Real = 24,
        line_spacing_height: Real | None = 1.0,
        justify: bool = False,
        indent: Real = 0.0,
        alignment: str | None = None,
        line_width: Real | None = None,
        font: str | None = "Consolas",
        slant: str = "NORMAL",
        weight: str = "NORMAL",
        base_color: ColorType = Color("white"),
        global_config: dict[str, str] | None = None,
        local_configs: dict[Selector, dict[str, str]] | None = None,
        disable_ligatures: bool = True,
        isolate: Selector = re.compile(r"\w+", flags=re.UNICODE),
        protect: Selector = (),
        width: Real | None = None,
        height: Real | None = None
    ):
        lexer = pygments.lexers.get_lexer_by_name(language)
        formatter = pygments.formatters.PangoMarkupFormatter(
            style=code_style
        )
        markup_string = pygments.highlight(code, lexer, formatter)
        markup_string = re.sub(r"</?tt>", "", markup_string)
        return super().__init__(
            string=markup_string,
            font_size=font_size,
            line_spacing_height=line_spacing_height,
            justify=justify,
            indent=indent,
            alignment=alignment,
            line_width=line_width,
            font=font,
            slant=slant,
            weight=weight,
            base_color=base_color,
            global_config=global_config,
            local_configs=local_configs,
            disable_ligatures=disable_ligatures,
            isolate=isolate,
            protect=protect,
            width=width,
            height=height
        )
