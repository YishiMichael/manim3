from __future__ import annotations


import os
import pathlib
import re
from typing import (
    ClassVar,
    Self,
    Unpack
)

import attrs

from ...constants.custom_typing import AlignmentType
from ...toplevel.toplevel import Toplevel
from .latex_string_mobject import (
    LatexStringMobjectIO,
    LatexStringMobjectInput,
    LatexStringMobjectKwargs
)
from .string_mobject import StringMobject


@attrs.frozen(kw_only=True)
class TexInput(LatexStringMobjectInput):
    alignment: AlignmentType = attrs.field(factory=lambda: Toplevel._get_config().tex_alignment)
    compiler: str = attrs.field(factory=lambda: Toplevel._get_config().tex_compiler)
    preambles: tuple[str, ...] = attrs.field(factory=lambda: Toplevel._get_config().tex_preambles)


class TexKwargs(LatexStringMobjectKwargs, total=False):
    alignment: AlignmentType
    compiler: str
    preambles: tuple[str, ...]


class TexIO[TexInputT: TexInput](LatexStringMobjectIO[TexInputT]):
    __slots__ = ()

    _dir_name: ClassVar[str] = "tex"
    _scale_factor_per_font_point: ClassVar[float] = 0.001577

    @classmethod
    def _create_svg(
        cls: type[Self],
        content: str,
        input_data: TexInputT,
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
            if os.system(" ".join((
                program,
                f"\"{tex_path}\"",
                "-interaction=batchmode",
                "-halt-on-error",
                f"-output-directory=\"{svg_path.parent}\"",
                ">", os.devnull
            ))) or os.system(" ".join((
                "dvisvgm",
                f"\"{svg_path.with_suffix(dvi_suffix)}\"",
                "-n",
                "-v", "0",
                "-o", f"\"{svg_path}\"",
                ">", os.devnull
            ))):
                error = OSError(f"TexIO: Failed to execute latex command")
                log_text = svg_path.with_suffix(".log").read_text(encoding="utf-8")
                if (error_match_obj := re.search(r"(?<=\n! ).*", log_text)) is not None:
                    error.add_note(f"LaTeX error: {error_match_obj.group()}")
                raise error

        finally:
            for suffix in (".tex", dvi_suffix, ".log", ".aux"):
                svg_path.with_suffix(suffix).unlink(missing_ok=True)


class Tex(StringMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[TexKwargs]
    ) -> None:
        super().__init__(TexIO.get(TexInput(string=string, **kwargs)))
