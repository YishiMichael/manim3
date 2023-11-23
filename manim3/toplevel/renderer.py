from __future__ import annotations


import pathlib
import subprocess
from typing import (
    IO,
    Iterator,
    Self
)

import numpy as np
from PIL import Image

from ..rendering.buffers.attributes_buffer import AttributesBuffer
from ..rendering.buffers.texture_buffer import TextureBuffer
from ..rendering.framebuffers.color_framebuffer import ColorFramebuffer
from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ..rendering.mgl_enums import PrimitiveMode
from ..rendering.vertex_array import VertexArray
from .toplevel import Toplevel
from .toplevel_resource import ToplevelResource


class VideoPipe:
    __slots__ = (
        "_writing_process",
        "_video_stream",
        "_writable"
    )

    def __init__(
        self: Self,
        video_path: pathlib.Path
    ) -> None:
        super().__init__()
        writing_process = subprocess.Popen((
            "ffmpeg",
            "-y",  # Overwrite output file if it exists.
            "-f", "rawvideo",
            "-s", f"{Toplevel._get_config().pixel_width}x{Toplevel._get_config().pixel_height}",
            "-pix_fmt", "rgb24",
            "-r", f"{Toplevel._get_config().fps}",
            "-i", "-",  # The input comes from a pipe.
            "-vf", "vflip",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-loglevel", "error",
            f"{video_path}"
        ), stdin=subprocess.PIPE)
        assert writing_process.stdin is not None
        self._writing_process: subprocess.Popen = writing_process
        self._video_stream: IO[bytes] = writing_process.stdin
        self._writable: bool = False

    def write(
        self: Self,
        data: bytes
    ) -> None:
        self._video_stream.write(data)

    def close(
        self: Self
    ) -> None:
        self._video_stream.close()
        self._writing_process.wait()
        self._writing_process.terminate()


class Renderer(ToplevelResource):
    __slots__ = (
        "_color_framebuffer",
        "_oit_framebuffer",
        "_oit_compose_vertex_array",
        "_livestream",
        "_video_pipes"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()

        samples = Toplevel._get_config().msaa_samples if Toplevel._get_config().use_msaa else 0
        color_framebuffer = ColorFramebuffer()
        oit_framebuffer = OITFramebuffer(samples=samples)
        oit_compose_vertex_array = VertexArray(
            shader_filename="oit_compose.glsl",
            texture_buffers=(
                TextureBuffer(
                    name="t_accum_map",
                    textures=oit_framebuffer._accum_texture
                ),
                TextureBuffer(
                    name="t_revealage_map",
                    textures=oit_framebuffer._revealage_texture
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
        self._livestream: bool = False
        self._video_pipes: dict[str, VideoPipe] = {}

    def __contextmanager__(
        self: Self
    ) -> Iterator[None]:
        Toplevel._renderer = self
        Toplevel._get_config().output_dir.mkdir(exist_ok=True)
        Toplevel._get_logger()._livestream = False
        Toplevel._get_logger()._recordings_count = 0
        yield
        Toplevel._get_logger()._recordings_count = None
        Toplevel._get_logger()._livestream = None
        for filename, video_pipe in self._video_pipes.items():
            video_pipe.close()
            Toplevel._get_logger().log(f"Recording saved to '{filename}'.")
        Toplevel._renderer = None

    def _render_frame(
        self: Self
    ) -> None:
        scene = Toplevel._get_scene()

        self._oit_framebuffer._msaa_framebuffer.clear()
        for mobject in scene._root_mobject.iter_descendants():
            mobject._render(self._oit_framebuffer)
        if self._oit_framebuffer._msaa_framebuffer is not self._oit_framebuffer._framebuffer:
            Toplevel._get_context().copy_framebuffer(
                dst=self._oit_framebuffer._framebuffer,
                src=self._oit_framebuffer._msaa_framebuffer
            )

        red, green, blue = scene._background_color
        alpha = scene._background_opacity
        self._color_framebuffer._framebuffer.clear(
            red=red, green=green, blue=blue, alpha=alpha
        )
        self._oit_compose_vertex_array.render(self._color_framebuffer)

    def process_frame(
        self: Self
    ) -> None:
        if Toplevel._get_window()._pyglet_window.context is None:
            # User has attempted to close the window.
            raise KeyboardInterrupt

        if self._livestream or self._video_pipes:
            self._render_frame()

        if self._livestream:
            Toplevel._get_context().blit_framebuffer(
                dst=Toplevel._get_context().screen_framebuffer,
                src=self._color_framebuffer._framebuffer
            )
            Toplevel._get_window()._pyglet_window.flip()

        if self._video_pipes:
            frame_data = self._color_framebuffer._color_texture.read()
            for video_pipe in self._video_pipes.values():
                if not video_pipe._writable:
                    continue
                video_pipe.write(frame_data)

    def start_livestream(
        self: Self
    ) -> None:
        if self._livestream:
            return
        self._livestream = True
        Toplevel._get_logger()._livestream = True
        Toplevel._get_logger().log("Start livestream.")

    def stop_livestream(
        self: Self
    ) -> None:
        if not self._livestream:
            return
        self._livestream = False
        Toplevel._get_logger()._livestream = False
        Toplevel._get_logger().log("Stop livestream.")

    def start_recording(
        self: Self,
        filename: str | None = None
    ) -> None:
        if filename is None:
            filename = f"{Toplevel._get_config().default_filename}.mp4"
        video_dir = Toplevel._get_config().video_output_dir
        video_dir.mkdir(exist_ok=True)
        video_path = video_dir.joinpath(filename)
        assert video_path.suffix == ".mp4", \
            f"Video format other than .mp4 is currently not supported: {video_path.suffix}"
        if (video_pipe := self._video_pipes.pop(filename, None)) is None:
            video_pipe = VideoPipe(video_path)
            self._video_pipes[filename] = video_pipe
        if video_pipe._writable:
            return
        video_pipe._writable = True
        assert (recordings_count := Toplevel._get_logger()._recordings_count) is not None
        Toplevel._get_logger()._recordings_count = recordings_count + 1
        Toplevel._get_logger().log(f"Start recording video to '{filename}'.")

    def stop_recording(
        self: Self,
        filename: str | None = None
    ) -> None:
        if filename is None:
            filename = f"{Toplevel._get_config().default_filename}.mp4"
        if (video_pipe := self._video_pipes.pop(filename, None)) is None:
            raise ValueError(f"Video pipe on {filename} not found.")
        if not video_pipe._writable:
            return
        video_pipe._writable = False
        assert (recordings_count := Toplevel._get_logger()._recordings_count) is not None
        Toplevel._get_logger()._recordings_count = recordings_count - 1
        Toplevel._get_logger().log(f"Stop recording video to '{filename}'.")

    def snapshot(
        self: Self,
        filename: str | None = None
    ) -> None:
        if filename is None:
            filename = f"{Toplevel._get_config().default_filename}.png"
        image_dir = Toplevel._get_config().image_output_dir
        image_dir.mkdir(exist_ok=True)
        image_path = image_dir.joinpath(filename)

        self._render_frame()
        Image.frombytes(
            "RGB",
            Toplevel._get_config().pixel_size,
            self._color_framebuffer._color_texture.read(),
            "raw"
        ).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(image_path)
        Toplevel._get_logger().log(f"Snapshot saved to '{filename}'.")
