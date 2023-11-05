from __future__ import annotations


import pathlib
from typing import (
    Self,
    Unpack
)

import attrs

from ...toplevel.toplevel import Toplevel
from .string_mobject import StringMobject
from .tex import (
    TexIO,
    TexInput,
    TexKwargs
)


@attrs.frozen(kw_only=True)
class MathTexInput(TexInput):
    inline: bool = attrs.field(factory=lambda: Toplevel.config.math_tex_inline)


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
            content = f"${content}$"
        else:
            content = f"\\begin{{align*}}\n{content}\n\\end{{align*}}"
        super()._create_svg(content, input_data, svg_path)


class MathTex(StringMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[MathTexKwargs]
    ) -> None:
        super().__init__(MathTexIO.get(MathTexInput(string=string, **kwargs)))
