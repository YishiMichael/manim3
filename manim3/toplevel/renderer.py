from __future__ import annotations


import pathlib
import subprocess
from typing import (
    IO,
    Iterator,
    Self
)

import ffmpeg
import numpy as np
from PIL import Image

from ..rendering.buffers.attributes_buffer import AttributesBuffer
from ..rendering.buffers.texture_buffer import TextureBuffer
from ..rendering.framebuffers.final_framebuffer import FinalFramebuffer
from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ..rendering.mgl_enums import PrimitiveMode
from ..rendering.vertex_array import VertexArray
from .toplevel import Toplevel
from .toplevel_resource import ToplevelResource


class VideoPipe:
    __slots__ = (
        "_video_path",
        "_video_stream",
        "_writing_process",
        "_writing"
    )

    def __init__(
        self: Self,
        video_path: pathlib.Path
    ) -> None:
        super().__init__()
        pixel_width, pixel_height = Toplevel._get_config().pixel_size
        writing_process = (
            ffmpeg
            .input(
                filename="pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{pixel_width}x{pixel_height}",
                framerate=Toplevel._get_config().fps
            )
            .vflip()
            .output(
                filename=video_path,
                pix_fmt="yuv420p",
                loglevel="quiet"
            )
            .run_async(
                pipe_stdin=True,
                overwrite_output=True
            )
        )
        self._video_path: pathlib.Path = video_path
        self._video_stream: IO[bytes] = writing_process.stdin
        self._writing_process: subprocess.Popen[bytes] = writing_process
        self._writing: bool = False

    @property
    def is_writing(
        self: Self
    ) -> bool:
        return self._writing

    def enable_writing(
        self: Self
    ) -> None:
        if self._writing:
            return
        self._writing = True
        Toplevel._get_logger().log(f"Start recording video to '{self._video_path}'.")

    def disable_writing(
        self: Self
    ) -> None:
        if not self._writing:
            return
        self._writing = False
        Toplevel._get_logger().log(f"Stop recording video to '{self._video_path}'.")

    def write(
        self: Self,
        data: bytes
    ) -> None:
        self._video_stream.write(data)

    def save(
        self: Self
    ) -> None:
        self._writing_process.communicate()
        Toplevel._get_logger().log(f"Recording saved to '{self._video_path}'.")


class Livestreamer:
    __slots__ = ("_livestreaming",)

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._livestreaming: bool = False

    @property
    def is_livestreaming(
        self: Self
    ) -> bool:
        return self._livestreaming

    def enable_livestreaming(
        self: Self
    ) -> None:
        if self._livestreaming:
            return
        self._livestreaming = True
        Toplevel._get_logger().log("Start livestreaming.")

    def disable_livestreaming(
        self: Self
    ) -> None:
        if not self._livestreaming:
            return
        self._livestreaming = False
        Toplevel._get_logger().log("Stop livestreaming.")

    def livestream_frame(
        self: Self,
        framebuffer: FinalFramebuffer
    ) -> None:
        Toplevel._get_context().blit_framebuffer(
            dst=Toplevel._get_context().screen_framebuffer,
            src=framebuffer._framebuffer
        )
        Toplevel._get_window()._pyglet_window.flip()


class VideoRecorder:
    __slots__ = (
        "_video_dir",
        "_video_pipes"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        video_dir = Toplevel._get_config().video_output_dir
        video_dir.mkdir(exist_ok=True)
        self._video_dir: pathlib.Path = video_dir
        self._video_pipes: dict[str, VideoPipe] = {}

    @property
    def is_recording(
        self: Self
    ) -> bool:
        return any(video_pipe.is_writing for video_pipe in self._video_pipes.values())

    def enable_recording(
        self: Self,
        filename: str
    ) -> None:
        if (video_pipe := self._video_pipes.get(filename)) is None:
            video_path = self._video_dir.joinpath(filename)
            assert video_path.suffix == ".mp4", \
                f"Video format other than .mp4 is currently not supported: {video_path.suffix}"
            video_pipe = VideoPipe(video_path)
            self._video_pipes[filename] = video_pipe
        video_pipe.enable_writing()

    def disable_recording(
        self: Self,
        filename: str
    ) -> None:
        if (video_pipe := self._video_pipes.get(filename)) is None:
            raise ValueError(f"Video pipe to {filename} not found.")
        video_pipe.disable_writing()

    def record_frame(
        self: Self,
        framebuffer: FinalFramebuffer
    ) -> None:
        frame_bytes = framebuffer._framebuffer.read()
        for video_pipe in self._video_pipes.values():
            if video_pipe.is_writing:
                video_pipe.write(frame_bytes)

    def save_videos(
        self: Self
    ) -> None:
        for video_pipe in self._video_pipes.values():
            video_pipe.save()


class ImageRecoder:
    __slots__ = ("_image_dir",)

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        image_dir = Toplevel._get_config().image_output_dir
        image_dir.mkdir(exist_ok=True)
        self._image_dir: pathlib.Path = image_dir

    def record_frame(
        self: Self,
        filename: str,
        framebuffer: FinalFramebuffer
    ) -> None:
        image_path = self._image_dir.joinpath(filename)
        Image.frombytes(
            "RGB",
            framebuffer._framebuffer.size,
            framebuffer._framebuffer.read(),
            "raw"
        ).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(image_path)
        Toplevel._get_logger().log(f"Snapshot saved to '{image_path}'.")


class Renderer(ToplevelResource):
    __slots__ = (
        "_final_framebuffer",
        "_oit_framebuffer",
        "_oit_compose_vertex_array",
        "_livestreamer",
        "_video_recorder",
        "_image_recoder"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()

        msaa_samples = Toplevel._get_config().msaa_samples
        final_framebuffer = FinalFramebuffer()
        oit_framebuffer = OITFramebuffer(samples=msaa_samples)
        oit_compose_vertex_array = VertexArray(
            shader_filename="oit_compose.glsl",
            custom_macros=(
                f"#define MSAA_SAMPLES {msaa_samples}",
            ),
            texture_buffers=(
                TextureBuffer(
                    name="t_accum_map",
                    textures=oit_framebuffer.get_attachment("accum")
                ),
                TextureBuffer(
                    name="t_revealage_map",
                    textures=oit_framebuffer.get_attachment("revealage")
                )
            ),
            attributes_buffer=AttributesBuffer(
                field_declarations=(
                    "vec2 in_coordinates",
                ),
                data_dict={
                    "in_coordinates": np.array((
                        (-1.0, -1.0),
                        (1.0, -1.0),
                        (1.0, 1.0),
                        (-1.0, 1.0)
                    ))
                },
                primitive_mode=PrimitiveMode.TRIANGLE_FAN,
                vertices_count=4
            )
        )

        self._final_framebuffer: FinalFramebuffer = final_framebuffer
        self._oit_framebuffer: OITFramebuffer = oit_framebuffer
        self._oit_compose_vertex_array: VertexArray = oit_compose_vertex_array
        self._livestreamer: Livestreamer = Livestreamer()
        self._video_recorder: VideoRecorder = VideoRecorder()
        self._image_recoder: ImageRecoder = ImageRecoder()

    def __contextmanager__(
        self: Self
    ) -> Iterator[None]:
        Toplevel._renderer = self
        Toplevel._get_config().output_dir.mkdir(exist_ok=True)
        yield
        self._video_recorder.save_videos()
        Toplevel._renderer = None

    def _render_frame(
        self: Self
    ) -> None:
        scene = Toplevel._get_scene()

        self._oit_framebuffer.clear()
        for mobject in scene._root_mobject.iter_descendants():
            for vertex_array in mobject._iter_vertex_arrays():
                self._oit_framebuffer.render(vertex_array)

        self._final_framebuffer.clear(color=(*scene._background_color, scene._background_opacity))
        self._final_framebuffer.render(self._oit_compose_vertex_array)

    def process_frame(
        self: Self
    ) -> None:
        if Toplevel._get_window()._pyglet_window.context is None:
            # User has attempted to close the window.
            raise KeyboardInterrupt
        if self._livestreamer.is_livestreaming or self._video_recorder.is_recording:
            self._render_frame()
        if self._livestreamer.is_livestreaming:
            self._livestreamer.livestream_frame(self._final_framebuffer)
        if self._video_recorder.is_recording:
            self._video_recorder.record_frame(self._final_framebuffer)

    def start_livestream(
        self: Self
    ) -> None:
        self._livestreamer.enable_livestreaming()

    def stop_livestream(
        self: Self
    ) -> None:
        self._livestreamer.disable_livestreaming()

    def start_recording(
        self: Self,
        filename: str | None
    ) -> None:
        if filename is None:
            filename = f"{Toplevel._get_config().default_filename}.mp4"
        self._video_recorder.enable_recording(filename)

    def stop_recording(
        self: Self,
        filename: str | None
    ) -> None:
        if filename is None:
            filename = f"{Toplevel._get_config().default_filename}.mp4"
        self._video_recorder.disable_recording(filename)

    def snapshot(
        self: Self,
        filename: str | None = None
    ) -> None:
        if filename is None:
            filename = f"{Toplevel._get_config().default_filename}.png"
        self._render_frame()
        self._image_recoder.record_frame(filename, self._final_framebuffer)
