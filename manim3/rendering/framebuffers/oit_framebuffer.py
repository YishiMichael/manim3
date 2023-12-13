from __future__ import annotations


from typing import Self

import moderngl

from ..mgl_enums import (
    BlendEquation,
    BlendFunc,
    ContextFlag
)
from .framebuffer import (
    Framebuffer,
    AttachmentInfo
)


class OITFramebuffer(Framebuffer):
    __slots__ = ()

    def __init__(
        self: Self,
        samples: int = 0
    ) -> None:
        super().__init__(
            attachment_info_dict={
                "accum": AttachmentInfo(
                    components=4,
                    dtype="f2",
                    src_blend_func=BlendFunc.ONE,
                    dst_blend_func=BlendFunc.ONE,
                    blend_equation=BlendEquation.FUNC_ADD
                ),
                "revealage": AttachmentInfo(
                    components=1,
                    dtype="f2",
                    src_blend_func=BlendFunc.ONE,
                    dst_blend_func=BlendFunc.ONE,
                    blend_equation=BlendEquation.FUNC_ADD
                )
            },
            samples=samples,
            flag=ContextFlag.BLEND
        )
