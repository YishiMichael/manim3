
import pathlib
from dataclasses import dataclass

from ...toplevel.toplevel import Toplevel
from .tex import (
    Tex,
    TexIO,
    TexInputData
)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class MathTexInputData(TexInputData):
    inline: bool


class MathTexIO(TexIO):
    __slots__ = ()

    @classmethod
    @property
    def _dir_name(cls) -> str:
        return "math_tex"

    @classmethod
    def _create_svg(
        cls,
        content: str,
        input_data: MathTexInputData,
        svg_path: pathlib.Path
    ) -> None:
        if input_data.inline:
            content = "".join(("$", content, "$"))
        else:
            content = "\n".join(("\\begin{align*}", content, "\\end{align*}"))
        super()._create_svg(content, input_data, svg_path)


class MathTex(Tex):
    __slots__ = ()

    def __init__(
        self,
        string: str,
        *,
        inline: bool | None = None,
        **kwargs
    ) -> None:
        config = Toplevel.config
        if inline is None:
            inline = config.math_tex_inline

        super().__init__(
            string=string,
            inline=inline,
            **kwargs
        )

    @classmethod
    @property
    def _io_cls(cls) -> type[MathTexIO]:
        return MathTexIO

    @classmethod
    @property
    def _input_data_cls(cls) -> type[MathTexInputData]:
        return MathTexInputData
