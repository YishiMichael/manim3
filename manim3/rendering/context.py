__all__ = [
    "Context",
    "ContextState"
]


#import atexit
from dataclasses import dataclass
from functools import reduce
import operator as op
import subprocess as sp
from typing import ClassVar

import moderngl
from moderngl_window.context.pyglet.window import Window
import OpenGL.GL as gl

from ..rendering.config import ConfigSingleton
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
    _MGL_CONTEXT: ClassVar[moderngl.Context | None] = None
    _WINDOW: ClassVar[Window | None] = None
    _WINDOW_FRAMEBUFFER: ClassVar[moderngl.Framebuffer | None] = None
    _WRITING_PROCESS: ClassVar[sp.Popen | None] = None

    def __new__(cls):
        raise TypeError

    @classmethod
    def activate(cls) -> None:
        if cls._MGL_CONTEXT is not None:
            return

        if ConfigSingleton().rendering.preview:
            window = Window(
                size=ConfigSingleton().size.window_pixel_size,
                fullscreen=False,
                resizable=True,
                gl_version=cls._GL_VERSION,
                vsync=True,
                cursor=True
            )
            mgl_context = window.ctx
            window_framebuffer = mgl_context.detect_framebuffer()
        else:
            window = None
            mgl_context = moderngl.create_context(
                require=cls._GL_VERSION[0] * 100 + cls._GL_VERSION[1] * 10,
                standalone=True
            )
            window_framebuffer = None
        mgl_context.gc_mode = "auto"
        #atexit.register(lambda: mgl_context.release())
        cls._MGL_CONTEXT = mgl_context
        cls._WINDOW = window
        cls._WINDOW_FRAMEBUFFER = window_framebuffer

    @classmethod
    def setup_writing_process(
        cls,
        scene_name: str
    ) -> None:
        cls._WRITING_PROCESS = sp.Popen((
            "ffmpeg",
            "-y",  # Overwrite output file if it exists.
            "-f", "rawvideo",
            "-s", "{}x{}".format(*ConfigSingleton().size.pixel_size),  # size of one frame
            "-pix_fmt", "rgba",
            "-r", str(ConfigSingleton().rendering.fps),  # frames per second
            "-i", "-",  # The input comes from a pipe.
            "-vf", "vflip",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            ConfigSingleton().path.output_dir.joinpath(f"{scene_name}.mp4")
        ), stdin=sp.PIPE)

    @classmethod
    @property
    def mgl_context(cls) -> moderngl.Context:
        assert (mgl_context := cls._MGL_CONTEXT) is not None
        return mgl_context

    @classmethod
    @property
    def window(cls) -> Window:
        assert (window := cls._WINDOW) is not None
        return window

    @classmethod
    @property
    def window_framebuffer(cls) -> moderngl.Framebuffer:
        assert (window_framebuffer := cls._WINDOW_FRAMEBUFFER) is not None
        return window_framebuffer

    @classmethod
    @property
    def writing_process(cls) -> sp.Popen:
        assert (writing_process := cls._WRITING_PROCESS) is not None
        return writing_process

    @classmethod
    def set_state(
        cls,
        context_state: ContextState
    ) -> None:
        context = Context.mgl_context
        context.enable_only(reduce(op.or_, (flag.value for flag in context_state.flags), ContextFlag.NOTHING.value))
        for index, (src_blend_func, dst_blend_func) in enumerate(context_state.blend_funcs):
            gl.glBlendFunci(
                index,
                src_blend_func.value,
                dst_blend_func.value
            )
        for index, blend_equation in enumerate(context_state.blend_equations):
            gl.glBlendEquationi(
                index,
                blend_equation.value
            )
        #context.blend_func = tuple(func.value for func in context_state.blend_func)
        #context.blend_equation = context_state.blend_equation.value
        context.depth_func = context_state.depth_func
        context.front_face = context_state.front_face
        context.cull_face = context_state.cull_face
        context.wireframe = context_state.wireframe

    @classmethod
    def texture(
        cls,
        *,
        size: tuple[int, int],
        components: int,
        dtype: str
    ) -> moderngl.Texture:
        texture = cls.mgl_context.texture(
            size=size,
            components=components,
            dtype=dtype
        )
        #atexit.register(lambda: texture.release())
        return texture

    @classmethod
    def depth_texture(
        cls,
        *,
        size: tuple[int, int]
    ) -> moderngl.Texture:
        depth_texture = cls.mgl_context.depth_texture(
            size=size
        )
        #atexit.register(lambda: depth_texture.release())
        return depth_texture

    @classmethod
    def framebuffer(
        cls,
        *,
        color_attachments: tuple[moderngl.Texture, ...],
        depth_attachment: moderngl.Texture | None
    ) -> moderngl.Framebuffer:
        framebuffer = cls.mgl_context.framebuffer(
            color_attachments=color_attachments,
            depth_attachment=depth_attachment
        )
        #atexit.register(lambda: framebuffer.release())
        return framebuffer

    @classmethod
    def buffer(
        cls,
        *,
        reserve: int = 1,
        dynamic: bool = True
    ) -> moderngl.Buffer:
        buffer = cls.mgl_context.buffer(reserve=reserve, dynamic=dynamic)  # TODO: dynamic?
        #atexit.register(lambda: buffer.release())
        return buffer

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
        program = cls.mgl_context.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
            geometry_shader=geometry_shader,
            tess_control_shader=tess_control_shader,
            tess_evaluation_shader=tess_evaluation_shader,
            varyings=varyings
        )
        #atexit.register(lambda: program.release())
        return program

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
        vertex_array = cls.mgl_context.vertex_array(
            program=program,
            content=content,
            index_buffer=index_buffer,
            mode=mode.value
        )
        #atexit.register(lambda: vertex_array.release())
        return vertex_array
