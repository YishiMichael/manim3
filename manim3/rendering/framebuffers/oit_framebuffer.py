from __future__ import annotations


from typing import Self

import moderngl

from ...lazy.lazy import Lazy
from ...toplevel.context import ContextState
from ...toplevel.toplevel import Toplevel
from ..mgl_enums import (
    BlendEquation,
    BlendFunc,
    ContextFlag
)
from .framebuffer import Framebuffer


class OITFramebuffer(Framebuffer):
    __slots__ = ()

    def __init__(
        self: Self,
        size: tuple[int, int] | None = None
    ) -> None:
        if size is None:
            size = Toplevel._get_config().pixel_size
        accum_texture = Toplevel._get_context().texture(
            size=size,
            components=4,
            dtype="f2"
        )
        revealage_texture = Toplevel._get_context().texture(
            size=size,
            components=1,
            dtype="f2"
        )
        super().__init__(
            color_attachments=(accum_texture, revealage_texture)
        )
        self._accum_texture_ = accum_texture
        self._revealage_texture_ = revealage_texture

    @Lazy.variable()
    @staticmethod
    def _accum_texture_() -> moderngl.Texture:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _revealage_texture_() -> moderngl.Texture:
        return NotImplemented

    @Lazy.property()
    @staticmethod
    def _context_state_() -> ContextState:
        return ContextState(
            flags=(ContextFlag.BLEND,),
            blend_funcs=((BlendFunc.ONE, BlendFunc.ONE), (BlendFunc.ONE, BlendFunc.ONE)),
            blend_equations=((BlendEquation.FUNC_ADD, BlendEquation.FUNC_ADD))
        )
