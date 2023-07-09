import moderngl

from ...config import Config
from ..context import (
    Context,
    ContextState
)
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
            size = Config().size.pixel_size  # rendering.texture_size = (2048, 2048)
        accum_texture = Context.texture(
            size=size,
            components=4,
            dtype="f2"
        )
        revealage_texture = Context.texture(
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
