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


class OITFramebuffer(Framebuffer):
    __slots__ = (
        "_accum_texture",
        "_revealage_texture"
    )

    def __init__(
        self: Self,
        samples: int = 0
    ) -> None:
        super().__init__(
            samples=samples,
            texture_infos={
                "accum": (4, "f2"),
                "revealage": (1, "f2")
            },
            context_state=ContextState(
                flags=(ContextFlag.BLEND,),
                blend_funcs=((BlendFunc.ONE, BlendFunc.ONE), (BlendFunc.ONE, BlendFunc.ONE)),
                blend_equations=((BlendEquation.FUNC_ADD, BlendEquation.FUNC_ADD))
            )
        )
        self._accum_texture: moderngl.Texture = self._named_textures["accum"]
        self._revealage_texture: moderngl.Texture = self._named_textures["revealage"]
