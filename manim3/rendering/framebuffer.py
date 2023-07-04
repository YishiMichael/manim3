import moderngl

from .context import (
    Context,
    ContextState
)
from .mgl_enums import (
    BlendEquation,
    BlendFunc,
    ContextFlag
)


class Framebuffer:
    __slots__ = (
        "framebuffer",
        "context_state"
    )

    def __init__(
        self,
        *,
        color_attachments: tuple[moderngl.Texture, ...] = (),
        depth_attachment: moderngl.Texture | None = None,
        framebuffer: moderngl.Framebuffer | None = None,
        context_state: ContextState
    ) -> None:
        if framebuffer is None:
            framebuffer = Context.framebuffer(
                color_attachments=color_attachments,
                depth_attachment=depth_attachment
            )
        self.framebuffer: moderngl.Framebuffer = framebuffer
        self.context_state: ContextState = context_state


class OITFramebuffer(Framebuffer):
    __slots__ = (
        "accum_texture",
        "revealage_texture"
    )

    def __init__(
        self,
        *,
        accum_texture: moderngl.Texture,
        revealage_texture: moderngl.Texture
    ) -> None:
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


class ColorFramebuffer(Framebuffer):
    __slots__ = ("color_texture",)

    def __init__(
        self,
        *,
        color_texture: moderngl.Texture
    ) -> None:
        super().__init__(
            color_attachments=(color_texture,),
            context_state=ContextState(
                flags=(ContextFlag.BLEND,)
            )
        )
        self.color_texture: moderngl.Texture = color_texture
