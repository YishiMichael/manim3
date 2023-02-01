__all__ = [
    "TexText",
    "Tex"
]


from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
import re
from typing import Generator
import warnings

from colour import Color
import toml

from ..config import Config
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
from ..utils.color import ColorUtils


@dataclass(
    frozen=True,
    kw_only=False,
    slots=True
)
class TexTemplate:
    description: str
    compiler: str
    preamble: str


def hash_string(string: str) -> str:
    # Truncating at 16 bytes for cleanliness
    hasher = hashlib.sha256(string.encode())
    return hasher.hexdigest()[:16]


@lru_cache(maxsize=8)
def get_tex_template(template_name: str | None) -> TexTemplate:
    default_name = "ctex"  # TODO: set global default
    if template_name is None:
        template_name = default_name
    name = template_name.replace(" ", "_").lower()
    with open(Config.tex_templates_path, encoding="utf-8") as tex_templates_file:
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
    content: str, template_name: str | None, additional_preamble: str | None
) -> str:
    template = get_tex_template(template_name)
    compiler = template.compiler
    preamble = template.preamble

    if additional_preamble is not None:
        preamble += "\n" + additional_preamble
    full_tex = "\n\n".join((
        "\\documentclass[preview]{standalone}",
        preamble,
        "\\begin{document}",
        content,
        "\\end{document}"
    )) + "\n"

    svg_file = os.path.join(
        Config.tex_dir, f"{hash_string(full_tex)}.svg"
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
    def __init__(
        self,
        string: str,
        *,
        font_size: Real = 48,
        alignment: str | None = "\\centering",
        tex_environment: str | None = None,
        template: str | None = None,
        additional_preamble: str | None = None,
        base_color: ColorType = Color("white"),
        tex_to_color_map: dict[str, ColorType] | None = None,
        isolate: Selector = (),
        protect: Selector = (),
        width: Real | None = None,
        height: Real | None = None
    ):
        # Prevent from passing an empty string.
        if not string.strip():
            string = "\\\\"
        if tex_to_color_map is None:
            tex_to_color_map = {}

        def get_content_prefix_and_suffix(is_labelled: bool) -> tuple[str, str]:
            prefix_lines: list[str] = []
            suffix_lines: list[str] = []
            if not is_labelled:
                color_hex = ColorUtils.color_to_hex(base_color)
                prefix_lines.append(self._get_color_command(
                    int(color_hex[1:], 16)
                ))
            if alignment is not None:
                prefix_lines.append(alignment)
            if tex_environment is not None:
                prefix_lines.append(f"\\begin{{{tex_environment}}}")
                suffix_lines.append(f"\\end{{{tex_environment}}}")
            return (
                "".join((line + "\n" for line in prefix_lines)),
                "".join(("\n" + line for line in suffix_lines))
            )

        def get_svg_path(content: str) -> str:
            with display_during_execution(f"Writing \"{string}\""):
                file_path = tex_content_to_svg_file(
                    content=content,
                    template_name=template,
                    additional_preamble=additional_preamble
                )
            return file_path

        super().__init__(
            string=string,
            isolate=isolate,
            protect=protect,
            configured_items_generator=(
                (span, {})
                for selector in tex_to_color_map
                for span in self._iter_spans_by_selector(selector, string)
            ),
            get_content_prefix_and_suffix=get_content_prefix_and_suffix,
            get_svg_path=get_svg_path,
            width=width,
            height=height,
            frame_scale=SCALE_FACTOR_PER_FONT_POINT * font_size
        )

        for selector, color in tex_to_color_map.items():
            self.select_parts(selector).set_fill(color=color)

    # Parsing

    @classmethod
    def _iter_command_matches(cls, string: str) -> Generator[re.Match[str], None, None]:
        # Lump together adjacent brace pairs
        pattern = re.compile(r"""
            (?P<command>\\(?:[a-zA-Z]+|.))
            |(?P<open>{+)
            |(?P<close>}+)
        """, flags=re.VERBOSE | re.DOTALL)

        def get_match_obj_by_span(span: tuple[int, int]) -> re.Match[str]:
            match_obj = pattern.fullmatch(string, pos=span[0], endpos=span[1])
            assert match_obj is not None
            return match_obj

        open_stack: list[tuple[int, int]] = []
        for match_obj in pattern.finditer(string):
            if not match_obj.group("close"):
                if not match_obj.group("open"):
                    yield match_obj
                    continue
                open_stack.append(match_obj.span())
                continue
            close_start, close_stop = match_obj.span()
            while True:
                if not open_stack:
                    raise ValueError("Missing '{' inserted")
                open_start, open_stop = open_stack.pop()
                n = min(open_stop - open_start, close_stop - close_start)
                yield get_match_obj_by_span((open_stop - n, open_stop))
                yield get_match_obj_by_span((close_start, close_start + n))
                close_start += n
                if close_start < close_stop:
                    continue
                open_stop -= n
                if open_start < open_stop:
                    open_stack.append((open_start, open_stop))
                break
        if open_stack:
            raise ValueError("Missing '}' inserted")

    @classmethod
    def _get_command_flag(cls, match_obj: re.Match[str]) -> CommandFlag:
        if match_obj.group("open"):
            return CommandFlag.OPEN
        if match_obj.group("close"):
            return CommandFlag.CLOSE
        return CommandFlag.OTHER

    @classmethod
    def _replace_for_content(cls, match_obj: re.Match[str]) -> str:
        return match_obj.group()

    @classmethod
    def _replace_for_matching(cls, match_obj: re.Match[str]) -> str:
        if match_obj.group("command"):
            return match_obj.group()
        return ""

    @classmethod
    def _get_attr_dict_from_command_pair(
        cls, open_command: re.Match[str], close_command: re.Match[str]
    ) -> dict[str, str] | None:
        if len(open_command.group()) >= 2:
            return {}
        return None

    @classmethod
    def _get_color_command(cls, rgb: int) -> str:
        rg, b = divmod(rgb, 256)
        r, g = divmod(rg, 256)
        return f"\\color[RGB]{{{r}, {g}, {b}}}"

    @classmethod
    def _get_command_string(
        cls, attr_dict: dict[str, str], edge_flag: EdgeFlag, label: int | None
    ) -> str:
        if label is None:
            return ""
        if edge_flag == EdgeFlag.STOP:
            return "}}"
        return "{{" + cls._get_color_command(label)


class Tex(TexText):
    def __init__(
        self,
        string: str,
        *,
        tex_environment: str | None = "align*",
        **kwargs
    ):
        super().__init__(string=string, tex_environment=tex_environment, **kwargs)
