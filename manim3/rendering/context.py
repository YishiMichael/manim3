__all__ = [
    "Context",
    "ContextState"
]


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
    def _mgl_context(cls) -> moderngl.Context:
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
        context = Context._mgl_context
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
        return cls._mgl_context.version_code

    @classmethod
    def texture(
        cls,
        *,
        size: tuple[int, int],
        components: int,
        dtype: str
    ) -> moderngl.Texture:
        return cls._mgl_context.texture(
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
        return cls._mgl_context.depth_texture(
            size=size
        )

    @classmethod
    def framebuffer(
        cls,
        *,
        color_attachments: tuple[moderngl.Texture, ...],
        depth_attachment: moderngl.Texture | None
    ) -> moderngl.Framebuffer:
        return cls._mgl_context.framebuffer(
            color_attachments=color_attachments,
            depth_attachment=depth_attachment
        )

    @classmethod
    def buffer(cls) -> moderngl.Buffer:
        # TODO: what effect does `dynamic` flag take?
        return cls._mgl_context.buffer(reserve=1, dynamic=True)

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
        return cls._mgl_context.program(
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
        return cls._mgl_context.vertex_array(
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
        return cls._mgl_context.scope(
            framebuffer=framebuffer,
            textures=textures,
            uniform_buffers=uniform_buffers
        )
