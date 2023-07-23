import moderngl

from ...toplevel.context import ContextState
from ...toplevel.toplevel import Toplevel
from ..mgl_enums import (
    BlendEquation,
    BlendFunc,
    ContextFlag
)
from .framebuffer import Framebuffer


class ColorFramebuffer(Framebuffer):
    __slots__ = ("color_texture",)

    def __init__(
        self,
        size: tuple[int, int] | None = None
    ) -> None:
        if size is None:
            size = Toplevel.config.pixel_size
        color_texture = Toplevel.context.texture(
            size=size,
            components=3,
            dtype="f1"
        )
        super().__init__(
            color_attachments=(color_texture,),
            context_state=ContextState(
                flags=(ContextFlag.BLEND,),
                blend_funcs=((BlendFunc.SRC_ALPHA, BlendFunc.ONE_MINUS_SRC_ALPHA),),
                blend_equations=(BlendEquation.FUNC_ADD,)
            )
        )
        self.color_texture: moderngl.Texture = color_texture
