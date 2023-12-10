from __future__ import annotations


from typing import Self

import attrs
import moderngl

from ...toplevel.toplevel import Toplevel
from ..mgl_enums import (
    BlendEquation,
    BlendFunc,
    ContextFlag
)


@attrs.frozen(kw_only=True)
class Texture_info:
    components: int
    dtype: str
    src_blend_func: BlendFunc
    dst_blend_func: BlendFunc
    blend_equation: BlendEquation


class Framebuffer:
    __slots__ = (
        "_named_textures",
        "_framebuffer",
        "_msaa_framebuffer",
        "_blendings",
        "_flag"
    )

    def __init__(
        self: Self,
        texture_info_dict: dict[str, Texture_info],
        samples: int,
        flag: ContextFlag
    ) -> None:
        super().__init__()
        size = Toplevel._get_config().pixel_size

        names: list[str] = []
        framebuffer_attachments: list[moderngl.Texture] = []
        msaa_framebuffer_attachments: list[moderngl.Texture] = []
        blendings: list[tuple[BlendFunc, BlendFunc, BlendEquation]] = []

        for name, texture_info in texture_info_dict.items():
            attachment = Toplevel._get_context().texture(
                size=size,
                components=texture_info.components,
                samples=0,
                dtype=texture_info.dtype
            )
            msaa_attachment = attachment if not samples else Toplevel._get_context().texture(
                size=size,
                components=texture_info.components,
                samples=samples,
                dtype=texture_info.dtype
            )

            names.append(name)
            framebuffer_attachments.append(attachment)
            msaa_framebuffer_attachments.append(msaa_attachment)
            blendings.append((texture_info.src_blend_func, texture_info.dst_blend_func, texture_info.blend_equation))

        self._named_textures: dict[str, moderngl.Texture] = dict(zip(names, framebuffer_attachments, strict=True))
        self._framebuffer: moderngl.Framebuffer = Toplevel._get_context().framebuffer(
            color_attachments=tuple(framebuffer_attachments)
        )
        self._msaa_framebuffer: moderngl.Framebuffer = Toplevel._get_context().framebuffer(
            color_attachments=tuple(msaa_framebuffer_attachments)
        )
        self._blendings: tuple[tuple[BlendFunc, BlendFunc, BlendEquation], ...] = tuple(blendings)
        self._flag: ContextFlag = flag

    def _render_msaa(
        self: Self,
        textures: tuple[tuple[moderngl.Texture, int], ...],
        uniform_buffers: tuple[tuple[moderngl.Buffer, int], ...],
        vertex_array: moderngl.VertexArray
    ) -> None:
        with Toplevel._get_context().scope(
            framebuffer=self._msaa_framebuffer,
            textures=textures,
            uniform_buffers=uniform_buffers
        ):
            Toplevel._get_context().set_blendings(self._blendings)
            Toplevel._get_context().set_flag(self._flag)
            vertex_array.render()
