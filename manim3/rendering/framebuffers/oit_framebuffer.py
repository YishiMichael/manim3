import moderngl

from ...toplevel.context import ContextState
from ...toplevel.toplevel import Toplevel
from ..mgl_enums import (
    BlendEquation,
    BlendFunc,
    ContextFlag
)
from .framebuffer import Framebuffer


class OITFramebuffer(Framebuffer):
    __slots__ = (
        "accum_texture",
        "revealage_texture"
    )

    def __init__(
        self,
        size: tuple[int, int] | None = None
    ) -> None:
        if size is None:
            size = Toplevel.config.pixel_size
        accum_texture = Toplevel.context.texture(
            size=size,
            components=4,
            dtype="f2"
        )
        revealage_texture = Toplevel.context.texture(
            size=size,
            components=1,
            dtype="f2"
        )
        super().__init__(
            color_attachments=(accum_texture, revealage_texture),
            context_state=ContextState(
                flags=(ContextFlag.BLEND,),
                blend_funcs=((BlendFunc.ONE, BlendFunc.ONE), (BlendFunc.ONE, BlendFunc.ONE)),
                blend_equations=((BlendEquation.FUNC_ADD, BlendEquation.FUNC_ADD))
            )
        )
        self.accum_texture: moderngl.Texture = accum_texture
        self.revealage_texture: moderngl.Texture = revealage_texture
