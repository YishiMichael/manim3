from __future__ import annotations


import pathlib
import subprocess
from types import TracebackType
from typing import (
    IO,
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
from .scene import Scene
from .toplevel import Toplevel


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
        "_livestream",
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
        self._livestream: bool = False
        self._video_pipes: dict[pathlib.Path, VideoPipe] = {}

    def __enter__(
        self: Self
    ) -> None:
        Toplevel._renderer = self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None
    ) -> None:
        for video_pipe in self._video_pipes.values():
            video_pipe.close()
        Toplevel._renderer = None

    def process_frame(
        self: Self,
        scene: Scene
    ) -> None:
        if self._livestream or self._video_pipes:
            self._oit_framebuffer._framebuffer_.clear()
            for mobject in scene._root_mobject.iter_descendants():
                mobject._render(self._oit_framebuffer)

            red, green, blue = scene._background_color
            alpha = scene._background_opacity
            self._color_framebuffer._framebuffer_.clear(
                red=red, green=green, blue=blue, alpha=alpha
            )
            self._oit_compose_vertex_array.render(self._color_framebuffer)

        if self._livestream:
            Toplevel._get_window().draw_frame(self._color_framebuffer._framebuffer_)
        if self._video_pipes:
            frame_data = self._color_framebuffer._color_texture_.read()
            for video_pipe in self._video_pipes.values():
                if not video_pipe._writable:
                    continue
                video_pipe.write(frame_data)

    def start_livestream(
        self: Self
    ) -> None:
        self._livestream = True

    def stop_livestream(
        self: Self
    ) -> None:
        self._livestream = False

    def start_recording(
        self: Self,
        filename: str | None = None
    ) -> None:
        if filename is None:
            video_path = Toplevel._get_config().video_output_dir.joinpath(Toplevel._get_config().default_filename).with_suffix(".mp4")
        else:
            video_path = Toplevel._get_config().video_output_dir.joinpath(filename)
            assert video_path.suffix == ".mp4", \
                f"Video format other than .mp4 is currently not supported: {video_path.suffix}"
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
            assert video_path.suffix == ".mp4", \
                f"Video format other than .mp4 is currently not supported: {video_path.suffix}"
        if (video_pipe := self._video_pipes.pop(video_path, None)) is None:
            raise ValueError(f"Video pipe on {filename} not found.")
        video_pipe.set_writability(False)

    def snapshot(
        self: Self,
        filename: str | None = None
    ) -> None:
        if filename is None:
            image_path = Toplevel._get_config().image_output_dir.joinpath(Toplevel._get_config().default_filename).with_suffix(".png")
        else:
            image_path = Toplevel._get_config().image_output_dir.joinpath(filename)
            assert image_path.suffix == ".png", \
                f"Image format other than .png is currently not supported: {image_path.suffix}"
        image = Image.frombytes(
            "RGB",
            Toplevel._get_config().pixel_size,
            self._color_framebuffer._color_texture_.read(),
            "raw"
        ).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image.save(image_path)
