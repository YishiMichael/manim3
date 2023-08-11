import os
import pathlib
import re
from dataclasses import dataclass
from typing import Iterable

from ...toplevel.toplevel import Toplevel
from ...utils.path_utils import PathUtils
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
class MathJaxInputData(LatexStringMobjectInputData):
    extensions: list[str]
    inline: bool


class MathJaxIO(LatexStringMobjectIO):
    __slots__ = ()

    @classmethod
    @property
    def _dir_name(cls) -> str:
        return "mathjax"

    @classmethod
    def _create_svg(
        cls,
        content: str,
        input_data: MathJaxInputData,
        svg_path: pathlib.Path
    ) -> None:
        mathjax_program_path = PathUtils.plugins_dir.joinpath("mathjax/index.js")
        full_content = content.replace("\n", " ")

        os.system(" ".join((
            "node",
            f"\"{mathjax_program_path}\"",
            f"--tex=\"{full_content}\"",
            f"--extensions=\"{' '.join(input_data.extensions)}\"",
            f"--inline={input_data.inline}",
            ">",
            f"\"{svg_path}\""
        )))
        svg_text = svg_path.read_text(encoding="utf-8")
        if (error_match_obj := re.search(r"<text\b.*?>(.*)</text>", svg_text)) is not None:
            raise IOError(f"MathJax error: {error_match_obj.group(1)}")

    @classmethod
    @property
    def _scale_factor_per_font_point(cls) -> float:
        return 0.009758


class MathJax(LatexStringMobject):
    __slots__ = ()

    def __init__(
        self,
        string: str,
        *,
        extensions: Iterable[str] | None = None,
        inline: bool | None = None,
        **kwargs
    ) -> None:
        config = Toplevel.config
        if extensions is None:
            extensions = config.mathjax_extensions
        if inline is None:
            inline = config.mathjax_inline

        super().__init__(
            string=string,
            extensions=list(extensions),
            inline=inline,
            **kwargs
        )

    @classmethod
    @property
    def _io_cls(cls) -> type[MathJaxIO]:
        return MathJaxIO

    @classmethod
    @property
    def _input_data_cls(cls) -> type[MathJaxInputData]:
        return MathJaxInputData
