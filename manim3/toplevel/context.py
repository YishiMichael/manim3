from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce
import operator as op
from typing import Iterator

import moderngl
import OpenGL.GL as gl

from ..rendering.mgl_enums import (
    BlendEquation,
    BlendFunc,
    ContextFlag,
    PrimitiveMode
)
from .config import Config


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ContextState:
    flags: tuple[ContextFlag, ...]
    blend_funcs: tuple[tuple[BlendFunc, BlendFunc], ...]
    blend_equations: tuple[BlendEquation, ...]


class Context:
    __slots__ = (
        "_mgl_context",
        "_window_framebuffer"
    )

    def __init__(
        self,
        mgl_context: moderngl.Context
    ) -> None:
        super().__init__()
        self._mgl_context: moderngl.Context = mgl_context
        self._window_framebuffer: moderngl.Framebuffer = mgl_context.detect_framebuffer()

    @classmethod
    @contextmanager
    def get_context(
        cls,
        config: Config
    ) -> "Iterator[Context]":
        mgl_context = moderngl.create_context(
            require=config.gl_version_code,
            standalone=not config.preview
        )
        yield Context(
            mgl_context=mgl_context
        )
        mgl_context.release()

    def set_state(
        self,
        context_state: ContextState
    ) -> None:
        self._mgl_context.enable_only(reduce(op.or_, (flag.value for flag in context_state.flags), ContextFlag.NOTHING.value))
        for index, ((src_blend_func, dst_blend_func), blend_equation) in enumerate(
            zip(context_state.blend_funcs, context_state.blend_equations, strict=True)
        ):
            gl.glBlendFunci(
                index,
                src_blend_func.value,
                dst_blend_func.value
            )
            gl.glBlendEquationi(
                index,
                blend_equation.value
            )

    def blit(
        self,
        src: moderngl.Framebuffer,
        dst: moderngl.Framebuffer
    ) -> None:
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src.glo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, dst.glo)
        gl.glBlitFramebuffer(
            *src.viewport, *dst.viewport,
            gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR
        )

    @property
    def version_code(self) -> int:
        return self._mgl_context.version_code

    @property
    def screen_framebuffer(self) -> moderngl.Framebuffer:
        return self._mgl_context.screen

    def texture(
        self,
        *,
        size: tuple[int, int],
        components: int,
        dtype: str
    ) -> moderngl.Texture:
        return self._mgl_context.texture(
            size=size,
            components=components,
            dtype=dtype
        )

    def depth_texture(
        self,
        *,
        size: tuple[int, int]
    ) -> moderngl.Texture:
        return self._mgl_context.depth_texture(
            size=size
        )

    def framebuffer(
        self,
        *,
        color_attachments: tuple[moderngl.Texture, ...],
        depth_attachment: moderngl.Texture | None
    ) -> moderngl.Framebuffer:
        return self._mgl_context.framebuffer(
            color_attachments=color_attachments,
            depth_attachment=depth_attachment
        )

    def buffer(self) -> moderngl.Buffer:
        # TODO: what effect does `dynamic` flag take?
        return self._mgl_context.buffer(reserve=1, dynamic=True)

    def program(
        self,
        *,
        vertex_shader: str,
        fragment_shader: str | None = None,
        geometry_shader: str | None = None,
        tess_control_shader: str | None = None,
        tess_evaluation_shader: str | None = None,
        varyings: tuple[str, ...] = ()
    ) -> moderngl.Program:
        return self._mgl_context.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
            geometry_shader=geometry_shader,
            tess_control_shader=tess_control_shader,
            tess_evaluation_shader=tess_evaluation_shader,
            varyings=varyings
        )

    def vertex_array(
        self,
        *,
        program: moderngl.Program,
        attributes_buffer: moderngl.Buffer,
        buffer_format_str: str,
        attribute_names: list[str],
        index_buffer: moderngl.Buffer | None,
        mode: PrimitiveMode
    ) -> moderngl.VertexArray:
        content = []
        if attribute_names:
            content.append((attributes_buffer, buffer_format_str, *attribute_names))
        return self._mgl_context.vertex_array(
            program=program,
            content=content,
            index_buffer=index_buffer,
            mode=mode.value
        )

    def scope(
        self,
        *,
        framebuffer: moderngl.Framebuffer | None = None,
        textures: tuple[tuple[moderngl.Texture, int], ...] = (),
        uniform_buffers: tuple[tuple[moderngl.Buffer, int], ...] = ()
    ) -> moderngl.Scope:
        return self._mgl_context.scope(
            framebuffer=framebuffer,
            textures=textures,
            uniform_buffers=uniform_buffers
        )
