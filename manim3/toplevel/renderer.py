from __future__ import annotations


import pathlib
import subprocess
from typing import (
    IO,
    Self
)

import numpy as np
import pyglet.gl as gl
from PIL import Image

from ..rendering.buffers.attributes_buffer import AttributesBuffer
from ..rendering.buffers.texture_buffer import TextureBuffer
from ..rendering.framebuffers.color_framebuffer import ColorFramebuffer
from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ..rendering.mgl_enums import PrimitiveMode
from ..rendering.vertex_array import VertexArray
from .scene import Scene
from .toplevel import Toplevel


class VideoPipe:
    __slots__ = (
        "_video_stream",
        "_writable"
    )

    def __init__(
        self: Self,
        video_path: pathlib.Path
    ) -> None:
        super().__init__()
        process = subprocess.Popen((
            "ffmpeg",
            "-y",  # Overwrite output file if it exists.
            "-f", "rawvideo",
            "-s", f"{Toplevel._get_config().pixel_width}x{Toplevel._get_config().pixel_height}",  # size of one frame
            "-pix_fmt", "rgb24",
            "-r", f"{Toplevel._get_config().fps}",  # frames per second
            "-i", "-",  # The input comes from a pipe.
            "-vf", "vflip",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            video_path  # TODO: str()
        ), stdin=subprocess.PIPE)
        assert process.stdin is not None
        self._video_stream: IO[bytes] = process.stdin
        self._writable: bool = False

    def write(
        self: Self,
        data: bytes
    ) -> None:
        self._video_stream.write(data)

    def set_writability(
        self: Self,
        writable: bool
    ) -> None:
        self._writable = writable


class Renderer:
    __slots__ = (
        "_color_framebuffer",
        "_oit_framebuffer",
        "_oit_compose_vertex_array",
        "_streaming",
        "_video_pipes"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()

        color_framebuffer = ColorFramebuffer()
        oit_framebuffer = OITFramebuffer()
        oit_compose_vertex_array = VertexArray(
            shader_filename="oit_compose.glsl",
            texture_buffers=(
                TextureBuffer(
                    name="t_accum_map",
                    textures=oit_framebuffer._accum_texture_
                ),
                TextureBuffer(
                    name="t_revealage_map",
                    textures=oit_framebuffer._revealage_texture_
                )
            ),
            attributes_buffer=AttributesBuffer(
                field_declarations=(
                    "vec3 in_position",
                    "vec2 in_uv"
                ),
                data_dict={
                    "in_position": np.array((
                        (-1.0, -1.0, 0.0),
                        (1.0, -1.0, 0.0),
                        (1.0, 1.0, 0.0),
                        (-1.0, 1.0, 0.0)
                    )),
                    "in_uv": np.array((
                        (0.0, 0.0),
                        (1.0, 0.0),
                        (1.0, 1.0),
                        (0.0, 1.0)
                    ))
                },
                primitive_mode=PrimitiveMode.TRIANGLE_FAN,
                vertices_count=4
            )
        )

        self._color_framebuffer: ColorFramebuffer = color_framebuffer
        self._oit_framebuffer: OITFramebuffer = oit_framebuffer
        self._oit_compose_vertex_array: VertexArray = oit_compose_vertex_array
        self._streaming: bool = False
        self._video_pipes: dict[pathlib.Path, VideoPipe] = {}

    def _process_frame(
        self: Self,
        scene: Scene
    ) -> None:
        if self._streaming or self._video_pipes:
            self._render_frame(scene)
        if self._streaming:
            self._stream_frame()
        if self._video_pipes:
            self._record_frame()

    def _render_frame(
        self: Self,
        scene: Scene
    ) -> None:
        self._oit_framebuffer._framebuffer_.clear()
        for mobject in scene._root_mobject.iter_descendants():
            mobject._render(self._oit_framebuffer)

        red, green, blue = scene._background_color
        alpha = scene._background_opacity
        self._color_framebuffer._framebuffer_.clear(
            red=red, green=green, blue=blue, alpha=alpha
        )
        self._oit_compose_vertex_array.render(self._color_framebuffer)

    def _stream_frame(
        self: Self
    ) -> None:
        src = self._color_framebuffer._framebuffer_
        dst = Toplevel._get_context().screen_framebuffer
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src.glo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, dst.glo)
        gl.glBlitFramebuffer(
            *src.viewport, *Toplevel._get_window().get_scene_viewport(),
            gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR
        )
        #Toplevel._get_context().blit(framebuffer, screen_framebuffer)
        Toplevel._get_window()._pyglet_window.flip()

    def _record_frame(
        self: Self
    ) -> None:
        frame_data = self._color_framebuffer._color_texture_.read()
        for video_pipe in self._video_pipes.values():
            if not video_pipe._writable:
                continue
            video_pipe.write(frame_data)

    def start_streaming(
        self: Self
    ) -> None:
        self._streaming = True

    def stop_streaming(
        self: Self
    ) -> None:
        self._streaming = False

    def start_recording(
        self: Self,
        filename: str | None = None
    ) -> None:
        if filename is None:
            video_path = Toplevel._get_config().video_output_dir.joinpath(Toplevel._get_config().default_filename).with_suffix(".mp4")
        else:
            video_path = Toplevel._get_config().video_output_dir.joinpath(filename)
            assert video_path.suffix == ".mp4", f"Video format other than .mp4 is currently not supported: {video_path.suffix}"
        if (video_pipe := self._video_pipes.pop(video_path, None)) is None:
            video_pipe = VideoPipe(video_path)
            self._video_pipes[video_path] = video_pipe
        video_pipe.set_writability(True)

    def stop_recording(
        self: Self,
        filename: str | None = None
    ) -> None:
        if filename is None:
            video_path = Toplevel._get_config().video_output_dir.joinpath(Toplevel._get_config().default_filename).with_suffix(".mp4")
        else:
            video_path = Toplevel._get_config().video_output_dir.joinpath(filename)
            assert video_path.suffix == ".mp4", f"Video format other than .mp4 is currently not supported: {video_path.suffix}"
        if (video_pipe := self._video_pipes.pop(video_path, None)) is None:
            raise ValueError(f"Video pipe on {filename} not found.")
        video_pipe.set_writability(False)

    def screenshot(
        self: Self,
        filename: str | None = None
    ) -> None:
        if filename is None:
            image_path = Toplevel._get_config().image_output_dir.joinpath(Toplevel._get_config().default_filename).with_suffix(".png")
        else:
            image_path = Toplevel._get_config().image_output_dir.joinpath(filename)
            assert image_path.suffix == ".png", f"Image format other than .png is currently not supported: {image_path.suffix}"
        image = Image.frombytes(
            "RGB",
            Toplevel._get_config().pixel_size,
            self._color_framebuffer._color_texture_.read(),
            "raw"
        ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image.save(image_path)
