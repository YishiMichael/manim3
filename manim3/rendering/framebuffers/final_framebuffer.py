from __future__ import annotations


from typing import Self

from ..mgl_enums import (
    BlendEquation,
    BlendFunc,
    ContextFlag
)
from .framebuffer import (
    Framebuffer,
    AttachmentInfo
)


class FinalFramebuffer(Framebuffer):
    __slots__ = ()

    def __init__(
        self: Self,
        samples: int = 0
    ) -> None:
        super().__init__(
            attachment_info_dict={
                "": AttachmentInfo(
                    components=4,
                    dtype="f1",
                    src_blend_func=BlendFunc.SRC_ALPHA,
                    dst_blend_func=BlendFunc.ONE_MINUS_SRC_ALPHA,
                    blend_equation=BlendEquation.FUNC_ADD
                )
            },
            samples=samples,
            flag=ContextFlag.BLEND
        )
