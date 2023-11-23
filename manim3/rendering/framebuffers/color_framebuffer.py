from __future__ import annotations


from typing import Self

import moderngl

from ...toplevel.context import ContextState
from ..mgl_enums import (
    BlendEquation,
    BlendFunc,
    ContextFlag
)
from .framebuffer import Framebuffer


class ColorFramebuffer(Framebuffer):
    __slots__ = ("_color_texture",)

    def __init__(
        self: Self,
        samples: int = 0
    ) -> None:
        super().__init__(
            samples=samples,
            texture_infos={
                "color": (3, "f1")
            },
            context_state=ContextState(
                flags=(ContextFlag.BLEND,),
                blend_funcs=((BlendFunc.SRC_ALPHA, BlendFunc.ONE_MINUS_SRC_ALPHA),),
                blend_equations=(BlendEquation.FUNC_ADD,)
            )
        )
        self._color_texture: moderngl.Texture = self._named_textures["color"]
