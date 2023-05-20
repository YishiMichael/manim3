from dataclasses import dataclass
from functools import reduce
import operator as op
from typing import ClassVar

import moderngl
from moderngl_window.context.pyglet.window import Window
import OpenGL.GL as gl

from ..config import ConfigSingleton
from ..rendering.mgl_enums import (
    BlendEquation,
    BlendFunc,
    ContextFlag,
    PrimitiveMode
)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ContextState:
    flags: tuple[ContextFlag, ...]
    blend_funcs: tuple[tuple[BlendFunc, BlendFunc], ...] = ((BlendFunc.SRC_ALPHA, BlendFunc.ONE_MINUS_SRC_ALPHA),)
    blend_equations: tuple[BlendEquation, ...] = (BlendEquation.FUNC_ADD,)
    depth_func: str = "<"
    front_face: str = "ccw"
    cull_face: str = "back"
    wireframe: bool = False


class Context:
    __slots__ = ()

    _GL_VERSION: ClassVar[tuple[int, int]] = (4, 3)
    _GL_VERSION_CODE: ClassVar[int] = 430

    _mgl_context: ClassVar[moderngl.Context | None] = None
    _window: ClassVar[Window | None] = None
    _window_framebuffer: ClassVar[moderngl.Framebuffer | None] = None

    def __new__(cls):
        raise TypeError

    @classmethod
    def activate(
        cls,
        title: str,
        standalone: bool
    ) -> None:
        assert cls._mgl_context is None
        if standalone:
            window = None
            mgl_context = moderngl.create_context(
                require=cls._GL_VERSION_CODE,
                standalone=True
            )
            window_framebuffer = None
        else:
            window = Window(
                title=title,
                size=ConfigSingleton().size.window_pixel_size,
                fullscreen=False,
                resizable=True,
                gl_version=cls._GL_VERSION,
                vsync=True,
                cursor=True
            )
            mgl_context = window.ctx
            window_framebuffer = mgl_context.detect_framebuffer()
        mgl_context.gc_mode = "auto"
        cls._mgl_context = mgl_context
        cls._window = window
        cls._window_framebuffer = window_framebuffer

    @classmethod
    @property
    def mgl_context(cls) -> moderngl.Context:
        assert (mgl_context := cls._mgl_context) is not None
        return mgl_context

    @classmethod
    @property
    def window(cls) -> Window:
        assert (window := cls._window) is not None
        return window

    @classmethod
    @property
    def window_framebuffer(cls) -> moderngl.Framebuffer:
        assert (window_framebuffer := cls._window_framebuffer) is not None
        return window_framebuffer

    @classmethod
    def set_state(
        cls,
        context_state: ContextState
    ) -> None:
        context = Context.mgl_context
        context.enable_only(reduce(op.or_, (flag.value for flag in context_state.flags), ContextFlag.NOTHING.value))
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
        context.depth_func = context_state.depth_func
        context.front_face = context_state.front_face
        context.cull_face = context_state.cull_face
        context.wireframe = context_state.wireframe

    @classmethod
    @property
    def version_code(cls) -> int:
        return cls.mgl_context.version_code

    @classmethod
    def texture(
        cls,
        *,
        size: tuple[int, int],
        components: int,
        dtype: str
    ) -> moderngl.Texture:
        return cls.mgl_context.texture(
            size=size,
            components=components,
            dtype=dtype
        )

    @classmethod
    def depth_texture(
        cls,
        *,
        size: tuple[int, int]
    ) -> moderngl.Texture:
        return cls.mgl_context.depth_texture(
            size=size
        )

    @classmethod
    def framebuffer(
        cls,
        *,
        color_attachments: tuple[moderngl.Texture, ...],
        depth_attachment: moderngl.Texture | None
    ) -> moderngl.Framebuffer:
        return cls.mgl_context.framebuffer(
            color_attachments=color_attachments,
            depth_attachment=depth_attachment
        )

    @classmethod
    def buffer(cls) -> moderngl.Buffer:
        # TODO: what effect does `dynamic` flag take?
        return cls.mgl_context.buffer(reserve=1, dynamic=True)

    @classmethod
    def program(
        cls,
        *,
        vertex_shader: str,
        fragment_shader: str | None = None,
        geometry_shader: str | None = None,
        tess_control_shader: str | None = None,
        tess_evaluation_shader: str | None = None,
        varyings: tuple[str, ...] = ()
    ) -> moderngl.Program:
        return cls.mgl_context.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
            geometry_shader=geometry_shader,
            tess_control_shader=tess_control_shader,
            tess_evaluation_shader=tess_evaluation_shader,
            varyings=varyings
        )

    @classmethod
    def vertex_array(
        cls,
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
        return cls.mgl_context.vertex_array(
            program=program,
            content=content,
            index_buffer=index_buffer,
            mode=mode.value
        )

    @classmethod
    def scope(
        cls,
        *,
        framebuffer: moderngl.Framebuffer | None = None,
        textures: tuple[tuple[moderngl.Texture, int], ...] = (),
        uniform_buffers: tuple[tuple[moderngl.Buffer, int], ...] = ()
    ) -> moderngl.Scope:
        return cls.mgl_context.scope(
            framebuffer=framebuffer,
            textures=textures,
            uniform_buffers=uniform_buffers
        )
