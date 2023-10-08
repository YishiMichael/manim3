from __future__ import annotations


import os
import pathlib
import re
from dataclasses import (
    dataclass,
    field
)
from typing import (
    Self,
    Unpack
)

from ...toplevel.toplevel import Toplevel
from ...utils.path_utils import PathUtils
from .latex_string_mobject import (
    LatexStringMobjectIO,
    LatexStringMobjectKwargs,
    LatexStringMobjectInput
)
from .string_mobject import StringMobject


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class MathJaxInput(LatexStringMobjectInput):
    extensions: tuple[str, ...] = field(default_factory=lambda: Toplevel.config.mathjax_extensions)
    inline: bool = field(default_factory=lambda: Toplevel.config.mathjax_inline)


class MathJaxKwargs(LatexStringMobjectKwargs, total=False):
    extensions: tuple[str, ...]
    inline: bool


class MathJaxIO[MathJaxInputT: MathJaxInput](LatexStringMobjectIO[MathJaxInputT]):
    __slots__ = ()

    @classmethod
    @property
    def _dir_name(
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
        mathjax_program_path = PathUtils.plugins_dir.joinpath("mathjax/index.js")
        full_content = content.replace("\n", " ")

        if os.system(" ".join((
            "node",
            f"\"{mathjax_program_path}\"",
            f"--tex=\"{full_content}\"",
            f"--extensions=\"{' '.join(input_data.extensions)}\"",
            f"--inline={input_data.inline}",
            f"--path=\"{svg_path}\"",
            ">", os.devnull
        ))):
            raise IOError("MathJaxIO: Failed to execute node command")
        svg_text = svg_path.read_text(encoding="utf-8")
        if (error_match_obj := re.search(r"<text\b.*?>(.*)</text>", svg_text)) is not None:
            raise ValueError(f"MathJax error: {error_match_obj.group(1)}")

    @classmethod
    @property
    def _scale_factor_per_font_point(
        cls: type[Self]
    ) -> float:
        return 0.009758


class MathJax(StringMobject):
    __slots__ = ()

    #_settings_dataclass: ClassVar[type[MathJaxInputSettings]] = MathJaxInputSettings
    #_io_cls: ClassVar[type[MathJaxIO]] = MathJaxIO

    def __init__(
        self: Self,
        string: str,
        #*,
        #extensions: list[str] | None = None,
        #inline: bool | None = None,
        **kwargs: Unpack[MathJaxKwargs]
    ) -> None:
        #config = Toplevel.config
        #if extensions is None:
        #    extensions = config.mathjax_extensions
        #if inline is None:
        #    inline = config.mathjax_inline

        super().__init__(MathJaxIO.get(MathJaxInput(string=string, **kwargs)))

    #@classmethod
    #@property
    #def _io_cls(
    #    cls: type[Self]
    #) -> type[MathJaxIO]:
    #    return MathJaxIO

    #@classmethod
    #@property
    #def _input_data_cls(
    #    cls: type[Self]
    #) -> type[MathJaxInputData]:
    #    return MathJaxInputData
