import moderngl

from ...lazy.lazy import (
    Lazy,
    LazyObject
)
from ...toplevel.context import ContextState
from ...toplevel.toplevel import Toplevel


class Framebuffer(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        *,
        color_attachments: tuple[moderngl.Texture, ...] = (),
        depth_attachment: moderngl.Texture | None = None
    ) -> None:
        super().__init__()
        self._framebuffer_ = Toplevel.context.framebuffer(
            color_attachments=color_attachments,
            depth_attachment=depth_attachment
        )

    @Lazy.variable()
    @staticmethod
    def _framebuffer_() -> moderngl.Framebuffer:
        return NotImplemented

    @Lazy.property()
    @staticmethod
    def _context_state_() -> ContextState:
        return NotImplemented
