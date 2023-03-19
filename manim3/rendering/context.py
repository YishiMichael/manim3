__all__ = [
    "Context",
    "ContextState"
]


from abc import (
    ABC,
    abstractmethod
)
import atexit
from dataclasses import dataclass
import os
import subprocess as sp
from typing import ClassVar

import moderngl
from moderngl_window.context.pyglet.window import Window

from ..rendering.config import ConfigSingleton


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ContextState:
    enable_only: int
    depth_func: str = "<"
    blend_func: tuple[int, int] | tuple[int, int, int, int] = moderngl.DEFAULT_BLENDING
    blend_equation: int | tuple[int, int] = moderngl.FUNC_ADD
    front_face: str = "ccw"
    cull_face: str = "back"
    wireframe: bool = False


class Context(ABC):
    __slots__ = ()

    _MGL_CONTEXT: ClassVar[moderngl.Context | None] = None
    _WINDOW: ClassVar[Window | None] = None
    _WINDOW_FRAMEBUFFER: ClassVar[moderngl.Framebuffer | None] = None
    _WRITING_PROCESS: ClassVar[sp.Popen | None] = None

    @abstractmethod
    def __new__(cls) -> None:
        pass

    @classmethod
    def activate(cls) -> None:
        if cls._MGL_CONTEXT is not None:
            return

        if ConfigSingleton().preview:
            window = Window(
                size=ConfigSingleton().window_pixel_size,
                fullscreen=False,
                resizable=True,
                gl_version=(3, 3),
                vsync=True,
                cursor=True
            )
            mgl_context = window.ctx
            window_framebuffer = mgl_context.detect_framebuffer()
        else:
            window = None
            mgl_context = moderngl.create_context(standalone=True)
            window_framebuffer = None
        atexit.register(lambda: mgl_context.release())
        cls._MGL_CONTEXT = mgl_context
        cls._WINDOW = window
        cls._WINDOW_FRAMEBUFFER = window_framebuffer

    @classmethod
    def setup_writing_process(
        cls,
        scene_name: str
    ) -> None:
        cls._WRITING_PROCESS = sp.Popen([
            "ffmpeg",
            "-y",  # Overwrite output file if it exists.
            "-f", "rawvideo",
            "-s", "{}x{}".format(*ConfigSingleton().pixel_size),  # size of one frame
            "-pix_fmt", "rgba",
            "-r", str(ConfigSingleton().fps),  # frames per second
            "-i", "-",  # The input comes from a pipe.
            "-vf", "vflip",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            os.path.join(ConfigSingleton().output_dir, f"{scene_name}.mp4")
        ], stdin=sp.PIPE)

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
        context.depth_func = context_state.depth_func
        context.blend_func = context_state.blend_func
        context.blend_equation = context_state.blend_equation
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
        atexit.register(lambda: texture.release())
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
        atexit.register(lambda: depth_texture.release())
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
        atexit.register(lambda: framebuffer.release())
        return framebuffer

    @classmethod
    def buffer(cls) -> moderngl.Buffer:
        buffer = cls.mgl_context.buffer(reserve=1, dynamic=True)  # TODO: dynamic?
        atexit.register(lambda: buffer.release())
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
    ) -> moderngl.Program:
        program = cls.mgl_context.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
            geometry_shader=geometry_shader,
            tess_control_shader=tess_control_shader,
            tess_evaluation_shader=tess_evaluation_shader
        )
        #atexit.register(lambda: program.release())
        return program

    @classmethod
    def vertex_array(
        cls,
        *,
        program: moderngl.Program,
        attributes_buffer: moderngl.Buffer,
        buffer_format: str,
        attribute_names: list[str],
        index_buffer: moderngl.Buffer,
        mode: int
    ) -> moderngl.VertexArray:
        vertex_array = cls.mgl_context.vertex_array(
            program=program,
            content=[(attributes_buffer, buffer_format, *attribute_names)],
            index_buffer=index_buffer,
            mode=mode
        )
        #atexit.register(lambda: vertex_array.release())
        return vertex_array
