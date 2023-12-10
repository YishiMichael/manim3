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


class OITFramebuffer(Framebuffer):
    __slots__ = ()

    def __init__(
        self: Self,
        samples: int = 0
    ) -> None:
        super().__init__(
            texture_info_dict={
                "accum": Texture_info(
                    components=4,
                    dtype="f2",
                    src_blend_func=BlendFunc.ONE,
                    dst_blend_func=BlendFunc.ONE,
                    blend_equation=BlendEquation.FUNC_ADD
                ),
                "revealage": Texture_info(
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

    @property
    def _accum_texture(
        self: Self
    ) -> moderngl.Texture:
        return self._named_textures["accum"]

    @property
    def _revealage_texture(
        self: Self
    ) -> moderngl.Texture:
        return self._named_textures["revealage"]
