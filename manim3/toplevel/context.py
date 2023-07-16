from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce
import operator as op
from typing import Iterator

import moderngl
#from moderngl_window.context.pyglet.window import Window
import OpenGL.GL as gl

from ..rendering.mgl_enums import (
    BlendEquation,
    BlendFunc,
    ContextFlag,
    PrimitiveMode
)
from .config import Config
#from .toplevel import Toplevel


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

    #_GL_VERSION_CODE: ClassVar[int] = 430

    def __init__(
        self,
        mgl_context: moderngl.Context
        #window: Window | None = None,
        #window_framebuffer: moderngl.Framebuffer | None = None
    ) -> None:
        super().__init__()
        self._mgl_context: moderngl.Context = mgl_context
        #self._window: Window | None = window
        self._window_framebuffer: moderngl.Framebuffer = mgl_context.detect_framebuffer()

    #mgl_context: ClassVar[moderngl.Context | None] = None
    #_window: ClassVar[Window | None] = None
    #_window_framebuffer: ClassVar[moderngl.Framebuffer | None] = None

    #@property
    #def window(self) -> Window:
    #    assert (window := self._window) is not None
    #    return window

    #@property
    #def window_framebuffer(self) -> moderngl.Framebuffer:
    #    assert (window_framebuffer := self._window_framebuffer) is not None
    #    return window_framebuffer

    #@classmethod
    #@contextmanager
    #def get_context(cls) -> "Iterator[Context]":
    #    mgl_context = moderngl.create_context(
    #        require=cls._GL_VERSION_CODE,
    #        standalone=True
    #    )
    #    yield Context(
    #        mgl_context=mgl_context
    #    )
    #    mgl_context.release()

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

    #@classmethod
    #def activate(
    #    cls,
    #    title: str,
    #    standalone: bool
    #) -> None:
    #    assert cls.mgl_context is None
    #    if standalone:
    #        window = None
    #        mgl_context = moderngl.create_context(
    #            require=cls._GL_VERSION_CODE,
    #            standalone=True
    #        )
    #        window_framebuffer = None
    #    else:
    #        window = Window(
    #            title=title,
    #            size=Config().size.window_pixel_size,
    #            fullscreen=False,
    #            resizable=True,
    #            gl_version=cls._GL_VERSION,
    #            vsync=True,
    #            cursor=True
    #        )
    #        mgl_context = window.ctx
    #        window_framebuffer = mgl_context.detect_framebuffer()

    #    mgl_context.gc_mode = "auto"
    #    cls.mgl_context = mgl_context
    #    cls._window = window
    #    cls._window_framebuffer = window_framebuffer

    #@classmethod
    #@property
    #def mgl_context(self) -> moderngl.Context:
    #    #assert (mgl_context := self._mgl_context) is not None
    #    return self._mgl_context

    #@classmethod
    #@property
    #def window(cls) -> Window:
    #    assert (window := cls._window) is not None
    #    return window

    #@classmethod
    #@property
    #def window_framebuffer(cls) -> moderngl.Framebuffer:
    #    assert (window_framebuffer := cls._window_framebuffer) is not None
    #    return window_framebuffer

    #@classmethod
    def set_state(
        self,
        context_state: ContextState
    ) -> None:
        #context = Context.mgl_context
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

    #@classmethod
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

    #@classmethod
    @property
    def version_code(self) -> int:
        return self._mgl_context.version_code

    @property
    def screen_framebuffer(self) -> moderngl.Framebuffer:
        return self._mgl_context.screen

    #@classmethod
    def texture(
        self,
        *,
        size: tuple[int, int],# | None = None,
        components: int,
        dtype: str
    ) -> moderngl.Texture:
        #if size is None:
        #    size = Config().size.pixel_size  # rendering.texture_size = (2048, 2048)
        return self._mgl_context.texture(
            size=size,
            components=components,
            dtype=dtype
        )

    #@classmethod
    def depth_texture(
        self,
        *,
        size: tuple[int, int]# | None = None
    ) -> moderngl.Texture:
        #if size is None:
        #    size = Config().size.pixel_size  # rendering.texture_size = (2048, 2048)
        return self._mgl_context.depth_texture(
            size=size
        )

    #@classmethod
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

    #@classmethod
    def buffer(self) -> moderngl.Buffer:
        # TODO: what effect does `dynamic` flag take?
        return self._mgl_context.buffer(reserve=1, dynamic=True)

    #@classmethod
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

    #@classmethod
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

    #@classmethod
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
