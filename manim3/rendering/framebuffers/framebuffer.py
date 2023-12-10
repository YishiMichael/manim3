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
from ..vertex_array import VertexArray


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
        "_use_msaa",
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

        use_msaa = bool(samples)
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
            msaa_attachment = attachment if not use_msaa else Toplevel._get_context().texture(
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
        self._use_msaa: bool = use_msaa
        self._flag: ContextFlag = flag

    def clear(
        self: Self,
        color: tuple[float, float, float, float] | None = None
    ) -> None:
        self._framebuffer.clear(color=color)
        if self._use_msaa:
            self._msaa_framebuffer.clear(color=color)

    def render_msaa(
        self: Self,
        vertex_array: VertexArray
    ) -> None:
        if (vertex_array_info := vertex_array._vertex_array_info_) is None:
            return None

        uniform_block_buffer_dict = {
            uniform_block_buffer._name_: uniform_block_buffer
            for uniform_block_buffer in vertex_array._uniform_block_buffers_
        }

        with Toplevel._get_context().scope(
            framebuffer=self._msaa_framebuffer,
            textures=vertex_array_info.texture_bindings,
            uniform_buffers=tuple(
                (uniform_block_buffer_dict[name]._buffer_, binding)
                for name, binding in vertex_array_info.uniform_block_bindings
            )
        ):
            Toplevel._get_context().set_blendings(self._blendings)
            Toplevel._get_context().set_flag(self._flag)
            vertex_array_info.vertex_array.render()

    def downsample_from_msaa(
        self: Self
    ) -> None:
        if self._use_msaa:
            Toplevel._get_context().copy_framebuffer(
                dst=self._framebuffer,
                src=self._msaa_framebuffer
            )
