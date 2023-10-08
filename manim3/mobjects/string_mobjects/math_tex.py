from __future__ import annotations


import pathlib
from dataclasses import (
    dataclass,
    field
)
from typing import (
    Self,
    Unpack
)

from ...toplevel.toplevel import Toplevel
from .string_mobject import StringMobject
from .tex import (
    TexIO,
    TexInput,
    TexKwargs
)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class MathTexInput(TexInput):
    inline: bool = field(default_factory=lambda: Toplevel.config.math_tex_inline)


class MathTexKwargs(TexKwargs, total=False):
    inline: bool


class MathTexIO[MathTexInputT: MathTexInput](TexIO[MathTexInputT]):
    __slots__ = ()

    @classmethod
    @property
    def _dir_name(
        cls: type[Self]
    ) -> str:
        return "math_tex"

    @classmethod
    def _create_svg(
        cls: type[Self],
        content: str,
        input_data: MathTexInputT,
        svg_path: pathlib.Path
    ) -> None:
        if input_data.inline:
            content = "".join(("$", content, "$"))
        else:
            content = "\n".join(("\\begin{align*}", content, "\\end{align*}"))
        super()._create_svg(content, input_data, svg_path)


class MathTex(StringMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[MathTexKwargs]
    ) -> None:
        super().__init__(MathTexIO.get(MathTexInput(string=string, **kwargs)))

    #@classmethod
    #@property
    #def _io_cls(
    #    cls: type[Self]
    #) -> type[MathTexIO]:
    #    return MathTexIO

    #@classmethod
    #@property
    #def _input_data_cls(
    #    cls: type[Self]
    #) -> type[MathTexInput]:
    #    return MathTexInput
