__all__ = ["Tex"]


from dataclasses import dataclass
from functools import lru_cache
import os
import pathlib
import re
from typing import (
    ClassVar,
    Generator
)

import toml

from ..custom_typing import (
    ColorType,
    Selector
)
from ..mobjects.string_mobject import (
    CommandFlag,
    EdgeFlag,
    StringFileWriter,
    StringMobject
)
from ..rendering.config import ConfigSingleton
from ..utils.color import ColorUtils


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TexTemplate:
    description: str
    compiler: str
    preamble: str


class TexFileWriter(StringFileWriter):
    __slots__ = ("_compiler",)

    def __init__(
        self,
        compiler: str
    ) -> None:
        super().__init__()
        self._compiler: str = compiler

    def get_svg_path(
        self,
        content: str
    ) -> pathlib.Path:
        hash_content = str((
            content,
            self._compiler
        ))
        return ConfigSingleton().path.tex_dir.joinpath(f"{self.hash_string(hash_content)}.svg")

    def create_svg_file(
        self,
        content: str,
        svg_path: pathlib.Path
    ) -> None:
        compiler = self._compiler
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

        # Write tex file.
        tex_path = svg_path.with_suffix(".tex")
        with tex_path.open(mode="w", encoding="utf-8") as tex_file:
            tex_file.write(content)

        try:
            # tex to dvi
            if os.system(" ".join((
                program,
                "-interaction=batchmode",
                "-halt-on-error",
                f"-output-directory=\"{svg_path.parent}\"",
                f"\"{tex_path}\"",
                ">",
                os.devnull
            ))):
                message = "LaTeX Error! Not a worry, it happens to the best of us."
                with svg_path.with_suffix(".log").open(encoding="utf-8") as log_file:
                    if (error_match_obj := re.search(r"(?<=\n! ).*", log_file.read())) is not None:
                        message += f" The error could be: `{error_match_obj.group()}`"
                raise ValueError(message)

            # dvi to svg
            os.system(" ".join((
                "dvisvgm",
                f"\"{svg_path.with_suffix(dvi_ext)}\"",
                "-n",
                "-v",
                "0",
                "-o",
                f"\"{svg_path}\"",
                ">",
                os.devnull
            )))

        finally:
            # Cleanup superfluous documents.
            for ext in (".tex", dvi_ext, ".log", ".aux"):
                svg_path.with_suffix(ext).unlink(missing_ok=True)


class Tex(StringMobject):
    __slots__ = ()

    TEX_SCALE_FACTOR_PER_FONT_POINT: ClassVar[float] = 0.001  # TODO

    def __init__(
        self,
        string: str,
        *,
        isolate: Selector = (),
        protect: Selector = (),
        tex_to_color_map: dict[str, ColorType] | None = None,
        template: str = ...,
        additional_preamble: str = ...,
        alignment: str | None = ...,
        environment: str | None = ...,
        base_color: ColorType = ...,
        font_size: float = ...
    ) -> None:
        # Prevent from passing an empty string.
        if not string.strip():
            string = "\\\\"
        if tex_to_color_map is None:
            tex_to_color_map = {}

        config = ConfigSingleton().tex
        if template is ...:
            template = config.template
        if additional_preamble is ...:
            additional_preamble = config.additional_preamble
        if alignment is ...:
            alignment = config.alignment
        if environment is ...:
            environment = config.environment
        if base_color is ...:
            base_color = config.base_color
        if font_size is ...:
            font_size = config.font_size

        tex_template = self._get_tex_templates_dict()[template]

        def get_content_by_body(
            body: str,
            is_labelled: bool
        ) -> str:
            prefix_lines: list[str] = []
            suffix_lines: list[str] = []
            if not is_labelled:
                color_hex = ColorUtils.color_to_hex(base_color)
                prefix_lines.append(self._get_color_command(
                    int(color_hex[1:], 16)
                ))
            if alignment is not None:
                prefix_lines.append(alignment)
            if environment is not None:
                prefix_lines.append(f"\\begin{{{environment}}}")
                suffix_lines.append(f"\\end{{{environment}}}")
            return "\n\n".join((
                "\\documentclass[preview]{standalone}",
                tex_template.preamble,
                additional_preamble,
                "\\begin{document}",
                "\n".join(prefix_lines),
                body,
                "\n".join(suffix_lines),
                "\\end{document}"
            )) + "\n"

        super().__init__(
            string=string,
            isolate=isolate,
            protect=protect,
            configured_items_generator=(
                (span, {})
                for selector in tex_to_color_map
                for span in self._iter_spans_by_selector(selector, string)
            ),
            get_content_by_body=get_content_by_body,
            file_writer=TexFileWriter(
                compiler=tex_template.compiler
            ),
            frame_scale=self.TEX_SCALE_FACTOR_PER_FONT_POINT * font_size
        )

        for selector, color in tex_to_color_map.items():
            self.select_parts(selector).set_fill(color=color)

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_tex_templates_dict() -> dict[str, TexTemplate]:
        with ConfigSingleton().path.tex_templates_path.open(encoding="utf-8") as tex_templates_file:
            template_content_dict = toml.load(tex_templates_file)
        return {
            name: TexTemplate(**template_content)
            for name, template_content in template_content_dict.items()
        }

    # Parsing

    @classmethod
    def _iter_command_matches(
        cls,
        string: str
    ) -> Generator[re.Match[str], None, None]:
        # Lump together adjacent brace pairs.
        pattern = re.compile(r"""
            (?P<command>\\(?:[a-zA-Z]+|.))
            |(?P<open>{+)
            |(?P<close>}+)
        """, flags=re.VERBOSE | re.DOTALL)

        def get_match_obj_by_span(
            span: tuple[int, int]
        ) -> re.Match[str]:
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
    def _get_command_flag(
        cls,
        match_obj: re.Match[str]
    ) -> CommandFlag:
        if match_obj.group("open"):
            return CommandFlag.OPEN
        if match_obj.group("close"):
            return CommandFlag.CLOSE
        return CommandFlag.OTHER

    @classmethod
    def _replace_for_content(
        cls,
        match_obj: re.Match[str]
    ) -> str:
        return match_obj.group()

    @classmethod
    def _replace_for_matching(
        cls,
        match_obj: re.Match[str]
    ) -> str:
        if match_obj.group("command"):
            return match_obj.group()
        return ""

    @classmethod
    def _get_attrs_from_command_pair(
        cls,
        open_command: re.Match[str],
        close_command: re.Match[str]
    ) -> dict[str, str] | None:
        if len(open_command.group()) >= 2:
            return {}
        return None

    @classmethod
    def _get_color_command(
        cls,
        rgb: int
    ) -> str:
        rg, b = divmod(rgb, 256)
        r, g = divmod(rg, 256)
        return f"\\color[RGB]{{{r}, {g}, {b}}}"

    @classmethod
    def _get_command_string(
        cls,
        attrs: dict[str, str],
        edge_flag: EdgeFlag,
        label: int | None
    ) -> str:
        if label is None:
            return ""
        if edge_flag == EdgeFlag.STOP:
            return "}}"
        return "{{" + cls._get_color_command(label)
