__all__ = [
    "TexText",
    "Tex"
]


from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
import re
from typing import Generator
import warnings

import toml

from ..constants import MANIM3_PATH
from ..custom_typing import (
    ColorType,
    Span
)
from ..mobjects.string_mobject import StringMobject


@dataclass
class TexTemplate:
    description: str
    compiler: str
    preamble: str


#SAVED_TEX_CONFIG = {}


def get_tex_dir() -> str:
    return "C:\\Users\\Michael\\AppData\\Local\\Temp\\Tex"  # TODO


def hash_string(string: str) -> str:
    # Truncating at 16 bytes for cleanliness
    hasher = hashlib.sha256(string.encode())
    return hasher.hexdigest()[:16]


def get_tex_template(template_name: str) -> TexTemplate:
    default_name = "ctex"  # TODO: set global default
    if not template_name:
        template_name = default_name
    name = template_name.replace(" ", "_").lower()
    with open(os.path.join(
        MANIM3_PATH, "tex_templates.tml"
    ), encoding="utf-8") as tex_templates_file:
        templates_dict = toml.load(tex_templates_file)
    if name not in templates_dict:
        warnings.warn(
            f"Cannot recognize template '{name}', falling back to '{default_name}'."
        )
        name = default_name
    return TexTemplate(**templates_dict[name])


#def get_tex_config() -> dict[str, str]:
#    """
#    Returns a dict which should look something like this:
#    {
#        "template": "default",
#        "compiler": "latex",
#        "preamble": "..."
#    }
#    """
#    # Only load once, then save thereafter
#    if not SAVED_TEX_CONFIG:
#        #template_name = get_custom_config()["style"]["tex_template"]
#        template_name = "ctex"
#        template = get_tex_template(template_name)
#        SAVED_TEX_CONFIG.update({
#            "template": template_name,
#            "compiler": template.compiler,
#            "preamble": template.preamble
#        })
#    return SAVED_TEX_CONFIG


def tex_content_to_svg_file(
    content: str, template_name: str, additional_preamble: str
) -> str:
    template = get_tex_template(template_name)
    compiler = template.compiler
    preamble = template.preamble

    if additional_preamble:
        preamble += "\n" + additional_preamble
    full_tex = "\n\n".join((
        "\\documentclass[preview]{standalone}",
        preamble,
        "\\begin{document}",
        content,
        "\\end{document}"
    )) + "\n"

    svg_file = os.path.join(
        get_tex_dir(), f"{hash_string(full_tex)}.svg"
    )
    if not os.path.exists(svg_file):
        # If svg doesn't exist, create it
        create_tex_svg(full_tex, svg_file, compiler)
    return svg_file


def create_tex_svg(full_tex: str, svg_file: str, compiler: str) -> None:
    if compiler == "latex":
        program = "latex"
        dvi_ext = ".dvi"
    elif compiler == "xelatex":
        program = "xelatex -no-pdf"
        dvi_ext = ".xdv"
    else:
        raise ValueError(
            f"Compiler '{compiler}' is not implemented"
        )

    # Write tex file
    root, _ = os.path.splitext(svg_file)
    with open(root + ".tex", "w", encoding="utf-8") as tex_file:
        tex_file.write(full_tex)

    # tex to dvi
    if os.system(" ".join((
        program,
        "-interaction=batchmode",
        "-halt-on-error",
        f"-output-directory=\"{os.path.dirname(svg_file)}\"",
        f"\"{root}.tex\"",
        ">",
        os.devnull
    ))):
        print("LaTeX Error! Not a worry, it happens to the best of us.")
        with open(root + ".log", "r", encoding="utf-8") as log_file:
            error_match_obj = re.search(r"(?<=\n! ).*", log_file.read())
            if error_match_obj:
                print(f"The error could be: `{error_match_obj.group()}`")
        raise LatexError()

    # dvi to svg
    os.system(" ".join((
        "dvisvgm",
        f"\"{root}{dvi_ext}\"",
        "-n",
        "-v",
        "0",
        "-o",
        f"\"{svg_file}\"",
        ">",
        os.devnull
    )))

    # Cleanup superfluous documents
    for ext in (".tex", dvi_ext, ".log", ".aux"):
        try:
            os.remove(root + ext)
        except FileNotFoundError:
            pass


# TODO, perhaps this should live elsewhere
@contextmanager
def display_during_execution(message: str) -> Generator[None, None, None]:
    # Merge into a single line
    to_print = message.replace("\n", " ")
    max_characters = 78
    if len(to_print) > max_characters:
        to_print = to_print[:max_characters - 3] + "..."
    try:
        print(to_print, end="\r")
        yield
    finally:
        print(" " * len(to_print), end="\r")


class LatexError(Exception):
    pass


SCALE_FACTOR_PER_FONT_POINT: float = 0.001


class TexText(StringMobject):
    #CONFIG = {
    #    "font_size": 48,
    #    "alignment": "\\centering",
    #    "tex_environment": "align*",
    #    "tex_to_color_map": {},
    #    "template": "",
    #    "additional_preamble": "",
    #}

    def __init__(
        self,
        string: str,
        *,
        #font_size: Real = 48,
        alignment: str = "\\centering",
        tex_environment: str | None = None,
        template: str = "",
        additional_preamble: str = "",
        tex_to_color_map: dict[str, ColorType] | None = None,
        **kwargs
    ):
        # Prevent from passing an empty string.
        if not string.strip():
            string = "\\\\"
        if tex_to_color_map is None:
            tex_to_color_map = {}
        #self.font_size: Real = font_size
        self.alignment: str = alignment
        self.tex_environment: str | None = tex_environment
        self.template: str = template
        self.additional_preamble: str = additional_preamble
        self.tex_to_color_map: dict[str, ColorType] = tex_to_color_map
        super().__init__(
            string=string,
            frame_scale=SCALE_FACTOR_PER_FONT_POINT * 48.0,
            **kwargs
        )

        for selector, color in tex_to_color_map.items():
            self.set_parts_color(selector, color)

    #@property
    #def hash_seed(self) -> tuple:
    #    return (
    #        self.__class__.__name__,
    #        self.svg_default,
    #        self.path_string_config,
    #        self.base_color,
    #        self.isolate,
    #        self.protect,
    #        self.tex_string,
    #        self.alignment,
    #        self.tex_environment,
    #        self.tex_to_color_map,
    #        self.template,
    #        self.additional_preamble
    #    )

    def get_file_path_by_content(self, content: str) -> str:
        with display_during_execution(f"Writing \"{self.string}\""):
            file_path = tex_content_to_svg_file(
                content, self.template, self.additional_preamble
            )
        return file_path

    # Parsing

    @classmethod
    def get_command_matches(cls, string: str) -> list[re.Match]:
        # Lump together adjacent brace pairs
        pattern = re.compile(r"""
            (?P<command>\\(?:[a-zA-Z]+|.))
            |(?P<open>{+)
            |(?P<close>}+)
        """, flags=re.X | re.S)
        result = []
        open_stack = []
        for match_obj in pattern.finditer(string):
            if match_obj.group("open"):
                open_stack.append((match_obj.span(), len(result)))
            elif match_obj.group("close"):
                close_start, close_end = match_obj.span()
                while True:
                    if not open_stack:
                        raise ValueError("Missing '{' inserted")
                    (open_start, open_end), index = open_stack.pop()
                    n = min(open_end - open_start, close_end - close_start)
                    result.insert(index, pattern.fullmatch(
                        string, pos=open_end - n, endpos=open_end
                    ))
                    result.append(pattern.fullmatch(
                        string, pos=close_start, endpos=close_start + n
                    ))
                    close_start += n
                    if close_start < close_end:
                        continue
                    open_end -= n
                    if open_start < open_end:
                        open_stack.append(((open_start, open_end), index))
                    break
            else:
                result.append(match_obj)
        if open_stack:
            raise ValueError("Missing '}' inserted")
        return result

    @classmethod
    def get_command_flag(cls, match_obj: re.Match) -> int:
        if match_obj.group("open"):
            return 1
        if match_obj.group("close"):
            return -1
        return 0

    @classmethod
    def replace_for_content(cls, match_obj: re.Match) -> str:
        return match_obj.group()

    @classmethod
    def replace_for_matching(cls, match_obj: re.Match) -> str:
        if match_obj.group("command"):
            return match_obj.group()
        return ""

    @classmethod
    def get_attr_dict_from_command_pair(
        cls, open_command: re.Match, close_command: re.Match
    ) -> dict[str, str] | None:
        if len(open_command.group()) >= 2:
            return {}
        return None

    @classmethod
    def get_color_command(cls, rgb: int) -> str:
        rg, b = divmod(rgb, 256)
        r, g = divmod(rg, 256)
        return f"\\color[RGB]{{{r}, {g}, {b}}}"

    @classmethod
    def get_command_string(
        cls, attr_dict: dict[str, str], is_end: bool, label: int | None
    ) -> str:
        if label is None:
            return ""
        if is_end:
            return "}}"
        return "{{" + cls.get_color_command(label)

    def get_configured_items(self) -> list[tuple[Span, dict[str, str]]]:
        return [
            (span, {})
            for selector in self.tex_to_color_map
            for span in self.find_spans_by_selector(selector, self.string)
        ]

    def get_content_prefix_and_suffix(
        self, is_labelled: bool
    ) -> tuple[str, str]:
        prefix_lines = []
        suffix_lines = []
        if not is_labelled:
            prefix_lines.append(self.get_color_command(
                self.color_to_int(self.base_color)
            ))
        if self.alignment:
            prefix_lines.append(self.alignment)
        if self.tex_environment:
            prefix_lines.append(f"\\begin{{{self.tex_environment}}}")
            suffix_lines.append(f"\\end{{{self.tex_environment}}}")
        return (
            "".join((line + "\n" for line in prefix_lines)),
            "".join(("\n" + line for line in suffix_lines))
        )

    # Method alias

    #def get_parts_by_tex(self, selector: Selector) -> VGroup:
    #    return self.select_parts(selector)

    #def get_part_by_tex(self, selector: Selector, **kwargs) -> VGroup:
    #    return self.select_part(selector, **kwargs)

    #def set_color_by_tex(self, selector: Selector, color: ManimColor):
    #    return self.set_parts_color(selector, color)

    #def set_color_by_tex_to_color_map(
    #    self, color_map: dict[Selector, ManimColor]
    #):
    #    return self.set_parts_color_by_dict(color_map)

    #def get_tex(self) -> str:
    #    return self.get_string()


class Tex(TexText):
    def __init__(
        self,
        string: str,
        tex_environment: str | None = "align*",
        **kwargs
    ):
        super().__init__(string=string, tex_environment=tex_environment, **kwargs)
