from __future__ import annotations


from typing import Self

import moderngl

from ...toplevel.context import ContextState
from ...toplevel.toplevel import Toplevel


class Framebuffer:
    __slots__ = (
        "_named_textures",
        "_msaa_framebuffer",
        "_framebuffer",
        "_context_state"
    )

    def __init__(
        self: Self,
        samples: int,
        texture_infos: dict[str, tuple[int, str]],
        context_state: ContextState
    ) -> None:
        super().__init__()
        size = Toplevel._get_config().pixel_size
        named_textures = {
            name: Toplevel._get_context().texture(
                size=size,
                components=components,
                samples=0,
                dtype=dtype
            )
            for name, (components, dtype) in texture_infos.items()
        }
        framebuffer = Toplevel._get_context().framebuffer(color_attachments=tuple(named_textures.values()))
        if samples:
            msaa_framebuffer = Toplevel._get_context().framebuffer(color_attachments=tuple(
                Toplevel._get_context().texture(
                    size=size,
                    components=color_attachment.components,
                    samples=samples,
                    dtype=color_attachment.dtype
                )
                for color_attachment in framebuffer.color_attachments
            ))
        else:
            msaa_framebuffer = framebuffer

        self._named_textures: dict[str, moderngl.Texture] = named_textures
        self._msaa_framebuffer: moderngl.Framebuffer = msaa_framebuffer
        self._framebuffer: moderngl.Framebuffer = framebuffer
        self._context_state: ContextState = context_state

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
            Toplevel._get_context().set_state(self._context_state)
            vertex_array.render()
