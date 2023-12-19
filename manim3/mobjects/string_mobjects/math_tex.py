#from __future__ import annotations


#from typing import (
#    Self,
#    Unpack
#)

#import attrs

#from ...constants.custom_typing import AlignmentType
#from ...toplevel.toplevel import Toplevel
#from .string_mobject import StringMobject
#from .tex import (
#    TexIO,
#    TexInput,
#    TexKwargs
#)


#@attrs.frozen(kw_only=True)
#class MathTexInput(TexInput):
#    alignment: AlignmentType = attrs.field(factory=lambda: Toplevel._get_config().math_tex_alignment)
#    inline: bool = attrs.field(factory=lambda: Toplevel._get_config().math_tex_inline)


#class MathTexKwargs(TexKwargs, total=False):
#    inline: bool


#class MathTexIO[MathTexInputT: MathTexInput](TexIO[MathTexInputT]):
#    __slots__ = ()

#    @classmethod
#    def _get_subdir_name(
#        cls: type[Self]
#    ) -> str:
#        return "math_tex"

#    @classmethod
#    def _get_environment_command_pair(
#        cls: type[Self],
#        input_data: MathTexInputT
#    ) -> tuple[str, str]:
#        if input_data.inline:
#            return "$", "$"
#        return "\\begin{align*}\n", "\n\\end{align*}"


#class MathTex(StringMobject):
#    __slots__ = ()

#    def __init__(
#        self: Self,
#        string: str,
#        **kwargs: Unpack[MathTexKwargs]
#    ) -> None:
#        super().__init__(MathTexIO.get(MathTexInput(string=string, **kwargs)))
