from abc import abstractmethod

from ..lazy.lazy import Lazy
from ..rendering.framebuffer import (
    OpaqueFramebuffer,
    TransparentFramebuffer
)
from ..rendering.gl_buffer import UniformBlockBuffer
from .mobject import (
    Mobject,
    MobjectStyleMeta
)


class RenderableMobject(Mobject):
    __slots__ = ()

    @MobjectStyleMeta.register()
    @Lazy.variable_hashable
    @classmethod
    def _is_transparent_(cls) -> bool:
        return False

    @Lazy.variable
    @classmethod
    def _camera_uniform_block_buffer_(cls) -> UniformBlockBuffer:
        # Keep updated with `Scene._camera._camera_uniform_block_buffer_`.
        return NotImplemented

    @abstractmethod
    def _render(
        self,
        target_framebuffer: OpaqueFramebuffer | TransparentFramebuffer
    ) -> None:
        pass
