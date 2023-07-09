import moderngl

from ...config import Config
from ..context import (
    Context,
    ContextState
)
from ..mgl_enums import ContextFlag
from .framebuffer import Framebuffer


class ColorFramebuffer(Framebuffer):
    __slots__ = ("color_texture",)

    def __init__(
        self,
        size: tuple[int, int] | None = None
    ) -> None:
        if size is None:
            size = Config().size.pixel_size  # rendering.texture_size = (2048, 2048)
        color_texture = Context.texture(
            size=size,
            components=3,
            dtype="f1"
        )
        super().__init__(
            color_attachments=(color_texture,),
            context_state=ContextState(
                flags=(ContextFlag.BLEND,)
            )
        )
        self.color_texture: moderngl.Texture = color_texture
