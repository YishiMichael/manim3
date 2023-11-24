from __future__ import annotations


import os
import pathlib
import re
from typing import (
    Self,
    Unpack
)

import attrs

from ...toplevel.toplevel import Toplevel
from .latex_string_mobject import (
    LatexStringMobjectIO,
    LatexStringMobjectInput,
    LatexStringMobjectKwargs
)
from .string_mobject import StringMobject


@attrs.frozen(kw_only=True)
class MathJaxInput(LatexStringMobjectInput):
    extensions: tuple[str, ...] = attrs.field(factory=lambda: Toplevel._get_config().mathjax_extensions)
    inline: bool = attrs.field(factory=lambda: Toplevel._get_config().mathjax_inline)


class MathJaxKwargs(LatexStringMobjectKwargs, total=False):
    extensions: tuple[str, ...]
    inline: bool


class MathJaxIO[MathJaxInputT: MathJaxInput](LatexStringMobjectIO[MathJaxInputT]):
    __slots__ = ()

    @classmethod
    def _get_subdir_name(
        cls: type[Self]
    ) -> str:
        return "mathjax"

    @classmethod
    def _create_svg(
        cls: type[Self],
        content: str,
        input_data: MathJaxInputT,
        svg_path: pathlib.Path
    ) -> None:
        mathjax_program_path = pathlib.Path(__import__("manim3").__file__).parent.joinpath("plugins/mathjax/index.js")
        full_content = content.replace("\n", " ")

        if os.system(" ".join((
            "node",
            f"\"{mathjax_program_path}\"",
            f"--tex=\"{full_content}\"",
            f"--extensions=\"{" ".join(input_data.extensions)}\"",
            f"--inline={input_data.inline}",
            f"--path=\"{svg_path}\"",
            ">", os.devnull
        ))):
            raise OSError("MathJaxIO: Failed to execute node command")
        svg_text = svg_path.read_text(encoding="utf-8")
        if (error_match_obj := re.search(r"<text\b.*?>(.*)</text>", svg_text)) is not None:
            error = OSError("MathJaxIO: Failed to execute node command")
            error.add_note(f"MathJax error: {error_match_obj.group(1)}")
            raise error

        # Seems svgelements cannot handle units in the root tag, so remove them manually.
        svg_path.write_text(re.sub(
            r"(width|height)=\"([\d\.]+)ex\"",
            lambda match: f"{match.group(1)}=\"{match.group(2)}\"",
            svg_path.read_text(encoding="utf-8")
        ), encoding="utf-8")

    @classmethod
    def _get_environment_command_pair(
        cls: type[Self],
        input_data: MathJaxInputT
    ) -> tuple[str, str]:
        return "{{", "}}"

    @classmethod
    def _get_scale_factor_per_font_point(
        cls: type[Self]
    ) -> float:
        return 0.4021


class MathJax(StringMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[MathJaxKwargs]
    ) -> None:
        super().__init__(MathJaxIO.get(MathJaxInput(string=string, **kwargs)))
