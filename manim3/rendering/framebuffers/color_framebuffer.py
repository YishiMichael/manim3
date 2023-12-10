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
    Texture_info
)


class ColorFramebuffer(Framebuffer):
    __slots__ = ()

    def __init__(
        self: Self,
        samples: int = 0
    ) -> None:
        super().__init__(
            texture_info_dict={
                "color": Texture_info(
                    components=3,
                    dtype="f1",
                    src_blend_func=BlendFunc.SRC_ALPHA,
                    dst_blend_func=BlendFunc.ONE_MINUS_SRC_ALPHA,
                    blend_equation=BlendEquation.FUNC_ADD
                )
            },
            samples=samples,
            flag=ContextFlag.BLEND
        )

    @property
    def _color_texture(
        self: Self
    ) -> moderngl.Texture:
        return self._named_textures["color"]
