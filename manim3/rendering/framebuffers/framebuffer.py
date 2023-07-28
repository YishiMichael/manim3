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

    @Lazy.variable_external
    @classmethod
    def _framebuffer_(cls) -> moderngl.Framebuffer:
        return NotImplemented

    @Lazy.property_external
    @classmethod
    def _context_state_(cls) -> ContextState:
        return NotImplemented
