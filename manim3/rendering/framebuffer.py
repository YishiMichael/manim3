__all__ = [
    "ColorFramebuffer",
    "Framebuffer",
    "TransparentFramebuffer",
    "OpaqueFramebuffer"
]


import moderngl

#import numpy as np

from ..rendering.context import (
    Context,
    ContextState
)
from ..rendering.mgl_enums import (
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


class OpaqueFramebuffer(Framebuffer):
    __slots__ = (
        "color_texture",
        "depth_texture"
    )

    def __init__(
        self,
        *,
        color_texture: moderngl.Texture,
        depth_texture: moderngl.Texture
    ) -> None:
        super().__init__(
            color_attachments=(color_texture,),
            depth_attachment=depth_texture,
            context_state=ContextState(
                flags=(ContextFlag.BLEND, ContextFlag.DEPTH_TEST),
                blend_funcs=((BlendFunc.ONE, BlendFunc.ZERO),)
            )
        )
        self.color_texture: moderngl.Texture = color_texture
        self.depth_texture: moderngl.Texture = depth_texture


class ColorFramebuffer(Framebuffer):
    __slots__ = ("color_texture",)

    def __init__(
        self,
        *,
        color_texture: moderngl.Texture
    ) -> None:
        super().__init__(
            color_attachments=(color_texture,),
            depth_attachment=None,
            context_state=ContextState(
                flags=()
            )
        )
        self.color_texture: moderngl.Texture = color_texture


class TransparentFramebuffer(Framebuffer):
    __slots__ = (
        "accum_texture",
        "revealage_texture",
        "depth_texture",
    )

    def __init__(
        self,
        *,
        accum_texture: moderngl.Texture,
        revealage_texture: moderngl.Texture,
        depth_texture: moderngl.Texture
    ) -> None:
        super().__init__(
            color_attachments=(accum_texture, revealage_texture),
            depth_attachment=depth_texture,
            context_state=ContextState(
                flags=(ContextFlag.BLEND, ContextFlag.DEPTH_TEST),
                blend_funcs=((BlendFunc.ONE, BlendFunc.ONE), (BlendFunc.ZERO, BlendFunc.ONE_MINUS_SRC_COLOR))
            )
        )
        self.accum_texture: moderngl.Texture = accum_texture
        self.revealage_texture: moderngl.Texture = revealage_texture
        self.depth_texture: moderngl.Texture = depth_texture

        #framebuffer = self.framebuffer
        ## Test against each fragment by the depth buffer, but never write to it.
        #framebuffer.depth_mask = False
        #framebuffer.clear()
        ## Initialize `revealage` with 1.0.
        ## TODO: There should be a more elegant way using `clear`.
        #revealage_texture.write(np.ones(revealage_texture.size, dtype="f2").tobytes())
