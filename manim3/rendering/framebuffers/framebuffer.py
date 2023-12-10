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
class AttachmentInfo:
    components: int
    dtype: str
    src_blend_func: BlendFunc
    dst_blend_func: BlendFunc
    blend_equation: BlendEquation


class Framebuffer:
    __slots__ = (
        "_named_attachments",
        "_framebuffer",
        "_blendings",
        "_flag"
    )

    def __init__(
        self: Self,
        attachment_info_dict: dict[str, AttachmentInfo],
        samples: int,
        flag: ContextFlag
    ) -> None:
        super().__init__()
        size = Toplevel._get_config().pixel_size
        attachment_items = tuple(
            (
                name,
                Toplevel._get_context().texture(
                    size=size,
                    components=attachment_info.components,
                    samples=samples,
                    dtype=attachment_info.dtype
                ),
                (attachment_info.src_blend_func, attachment_info.dst_blend_func, attachment_info.blend_equation)
            )
            for name, attachment_info in attachment_info_dict.items()
        )

        self._named_attachments: dict[str, moderngl.Texture] = {
            name: attachment for name, attachment, _ in attachment_items
        }
        self._framebuffer: moderngl.Framebuffer = Toplevel._get_context().framebuffer(color_attachments=tuple(
            attachment for _, attachment, _ in attachment_items
        ))
        self._blendings: tuple[tuple[BlendFunc, BlendFunc, BlendEquation], ...] = tuple(
            blending for _, _, blending in attachment_items
        )
        self._flag: ContextFlag = flag

    def clear(
        self: Self,
        color: tuple[float, float, float, float] | None = None
    ) -> None:
        self._framebuffer.clear(color=color)

    def render(
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
            framebuffer=self._framebuffer,
            textures=vertex_array_info.texture_bindings,
            uniform_buffers=tuple(
                (uniform_block_buffer_dict[name]._buffer_, binding)
                for name, binding in vertex_array_info.uniform_block_bindings
            )
        ):
            Toplevel._get_context().set_blendings(self._blendings)
            Toplevel._get_context().set_flag(self._flag)
            vertex_array_info.vertex_array.render()

    def copy_from(
        self: Self,
        framebuffer: Framebuffer
    ) -> None:
        Toplevel._get_context().copy_framebuffer(
            dst=self._framebuffer,
            src=framebuffer._framebuffer
        )
