from dataclasses import dataclass
from functools import lru_cache
import os
import pathlib
import re
from typing import (
    ClassVar,
    Iterable,
    Iterator
)

import toml

from ...config import Config
from ...constants import Alignment
from ...custom_typing import (
    ColorT,
    SelectorT
)
from .string_mobject import (
    CommandFlag,
    EdgeFlag,
    StringFileWriter,
    StringMobject,
    StringParser
)


class LaTeXError(ValueError):
    def __init__(
        self,
        error_message: str | None
    ) -> None:
        message = "LaTeX Error! Not a worry, it happens to the best of us."
        if error_message is not None:
            message += f" The error could be: `{error_message}`"
        super().__init__(message)


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
    __slots__ = ()

    _DIR_NAME: ClassVar[str] = "_tex"

    @classmethod
    def create_svg_file(
        cls,
        content: str,
        svg_path: pathlib.Path,
        preamble: str,
        template: str,
        alignment: Alignment,
        environment: str
    ) -> None:
        tex_template = cls._get_tex_templates_dict()[template]
        compiler = tex_template.compiler
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
            match alignment:
                case Alignment.LEFT:
                    alignment_command = "\\flushleft"
                case Alignment.CENTER:
                    alignment_command = "\\centering"
                case Alignment.RIGHT:
                    alignment_command = "\\flushright"
            if environment:
                begin_environment = f"\\begin{{{environment}}}"
                end_environment = f"\\end{{{environment}}}"
            else:
                begin_environment = ""
                end_environment = ""
            full_content = "\n".join((
                preamble,
                tex_template.preamble,
                "\\begin{document}",
                alignment_command,
                begin_environment,
                content,
                end_environment,
                "\\end{document}"
            )) + "\n"
            tex_file.write(full_content)

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
                error_message: str | None = None
                with svg_path.with_suffix(".log").open(encoding="utf-8") as log_file:
                    if (error_match_obj := re.search(r"(?<=\n! ).*", log_file.read())) is not None:
                        error_message = error_match_obj.group()
                raise LaTeXError(error_message)

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

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_tex_templates_dict() -> dict[str, TexTemplate]:
        with Config().path.tex_templates_path.open(encoding="utf-8") as tex_templates_file:
            template_content_dict = toml.load(tex_templates_file)
        return {
            name: TexTemplate(**template_content)
            for name, template_content in template_content_dict.items()
        }


class MathjaxFileWriter(StringFileWriter):
    __slots__ = ()

    _DIR_NAME: ClassVar[str] = "_mathjax"

    @classmethod
    def create_svg_file(
        cls,
        content: str,
        svg_path: pathlib.Path
    ) -> None:
        import manimgl_mathjax  # TODO
        mathjax_program_path = pathlib.Path(manimgl_mathjax.__file__).absolute().with_name("index.js")

        try:
            full_content = content.replace("\n", " ")
            os.system(" ".join((
                "node",
                f"\"{mathjax_program_path}\"",
                f"\"{svg_path}\"",
                f"\"{full_content}\"",
                ">",
                os.devnull
            )))
            with svg_path.open(encoding="utf-8") as svg_file:
                if (error_match_obj := re.search(r"(?<=data\-mjx\-error\=\")(.*?)(?=\")", svg_file.read())) is not None:
                    raise LaTeXError(error_match_obj.group())

        except LaTeXError:
            svg_path.unlink()
            raise


class TexParser(StringParser):
    __slots__ = ()

    @classmethod
    def _iter_command_matches(
        cls,
        string: str
    ) -> Iterator[re.Match[str]]:
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
    def _get_command_string(
        cls,
        label: int | None,
        edge_flag: EdgeFlag,
        attrs: dict[str, str]
    ) -> str:
        if label is None:
            return ""
        if edge_flag == EdgeFlag.STOP:
            return "}}"
        rg, b = divmod(label, 256)
        r, g = divmod(rg, 256)
        color_command = f"\\color[RGB]{{{r}, {g}, {b}}}"
        return "{{" + color_command


class Tex(StringMobject):
    __slots__ = ()

    # In this convension, `font_size=30` would make the height of "x" become roughly 0.30.
    _TEX_SCALE_FACTOR_PER_FONT_POINT: ClassVar[float] = 0.001577
    _MATHJAX_SCALE_FACTOR: ClassVar[float] = 6.188

    def __init__(
        self,
        string: str,
        *,
        isolate: Iterable[SelectorT] = (),
        protect: Iterable[SelectorT] = (),
        tex_to_color_map: dict[SelectorT, ColorT] | None = None,
        use_mathjax: bool | None = None,
        preamble: str | None = None,
        template: str | None = None,
        alignment: Alignment | None = None,
        environment: str | None = None,
        base_color: ColorT | None = None,
        font_size: float | None = None
    ) -> None:
        # Prevent from passing an empty string.
        if not string.strip():
            string = "\\\\"
        if tex_to_color_map is None:
            tex_to_color_map = {}

        config = Config().tex
        if use_mathjax is None:
            use_mathjax = config.use_mathjax
        if preamble is None:
            preamble = config.preamble
        if template is None:
            template = config.template
        if alignment is None:
            alignment = config.alignment
        if environment is None:
            environment = config.environment
        if base_color is None:
            base_color = config.base_color
        if font_size is None:
            font_size = config.font_size

        frame_scale = font_size * self._TEX_SCALE_FACTOR_PER_FONT_POINT

        if not use_mathjax:
            file_writer = TexFileWriter(
                preamble=preamble,
                template=template,
                alignment=alignment,
                environment=environment
            )
        else:
            frame_scale *= self._MATHJAX_SCALE_FACTOR
            # `template`, `preamble`, `alignment`, `environment`
            # all don't make an effect when using mathjax.
            file_writer = MathjaxFileWriter()

        parser = TexParser(
            string=string,
            isolate=isolate,
            protect=protect,
            local_attrs={
                selector: {}
                for selector in tex_to_color_map
            },
            global_attrs={},
            file_writer=file_writer,
            frame_scale=frame_scale
        )
        super().__init__(
            parser=parser
        )

        self.set_style(color=base_color)
        for selector, color in tex_to_color_map.items():
            self.select_parts(selector).set_style(color=color)
