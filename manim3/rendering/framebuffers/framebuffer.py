import moderngl

from ..context import (
    Context,
    ContextState
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
