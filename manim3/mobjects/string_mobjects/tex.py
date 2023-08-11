import os
import pathlib
import re
from dataclasses import dataclass
from typing import Iterable

from ...constants.custom_typing import AlignmentT
from ...toplevel.toplevel import Toplevel
from .latex_string_mobject import (
    LatexStringMobject,
    LatexStringMobjectIO,
    LatexStringMobjectInputData
)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TexInputData(LatexStringMobjectInputData):
    compiler: str
    preambles: list[str]
    alignment: AlignmentT


class TexIO(LatexStringMobjectIO):
    __slots__ = ()

    @classmethod
    @property
    def _dir_name(cls) -> str:
        return "tex"

    @classmethod
    def _create_svg(
        cls,
        content: str,
        input_data: TexInputData,
        svg_path: pathlib.Path
    ) -> None:
        match input_data.compiler:
            case "latex":
                program = "latex"
                dvi_suffix = ".dvi"
            case "xelatex":
                program = "xelatex -no-pdf"
                dvi_suffix = ".xdv"
            case _:
                raise ValueError(f"Compiler '{input_data.compiler}' is not implemented")

        match input_data.alignment:
            case "left":
                alignment_command = "\\flushleft"
            case "center":
                alignment_command = "\\centering"
            case "right":
                alignment_command = "\\flushright"

        full_content = "\n".join((
            "\\documentclass[preview]{standalone}",
            *input_data.preambles,
            "\\begin{document}",
            alignment_command,
            content,
            "\\end{document}"
        )) + "\n"

        tex_path = svg_path.with_suffix(".tex")
        tex_path.write_text(full_content, encoding="utf-8")

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
                error_message = "LaTeX error"
                log_text = svg_path.with_suffix(".log").read_text(encoding="utf-8")
                if (error_match_obj := re.search(r"(?<=\n! ).*", log_text)) is not None:
                    error_message += f": {error_match_obj.group()}"
                raise IOError(error_message)

            # dvi to svg
            os.system(" ".join((
                "dvisvgm",
                f"\"{svg_path.with_suffix(dvi_suffix)}\"",
                "-n",
                "-v",
                "0",
                "-o",
                f"\"{svg_path}\"",
                ">",
                os.devnull
            )))

        finally:
            for suffix in (".tex", dvi_suffix, ".log", ".aux"):
                svg_path.with_suffix(suffix).unlink(missing_ok=True)

    @classmethod
    @property
    def _scale_factor_per_font_point(cls) -> float:
        return 0.001577  # TODO: Affected by frame height?


class Tex(LatexStringMobject):
    __slots__ = ()

    def __init__(
        self,
        string: str,
        *,
        alignment: AlignmentT | None = None,
        compiler: str | None = None,
        preambles: Iterable[str] | None = None,
        **kwargs
    ) -> None:
        config = Toplevel.config
        if alignment is None:
            alignment = config.tex_alignment
        if compiler is None:
            compiler = config.tex_compiler
        if preambles is None:
            preambles = config.tex_preambles

        super().__init__(
            string=string,
            alignment=alignment,
            compiler=compiler,
            preambles=list(preambles),
            **kwargs
        )

    @classmethod
    @property
    def _io_cls(cls) -> type[TexIO]:
        return TexIO

    @classmethod
    @property
    def _input_data_cls(cls) -> type[TexInputData]:
        return TexInputData
