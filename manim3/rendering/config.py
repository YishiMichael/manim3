__all__ = [
    "Config",
    "ConfigSingleton"
]


from abc import ABC
import pathlib
import sys
from typing import ClassVar


class Config(ABC):
    __slots__ = (
        "_camera_altitude",
        "_camera_near",
        "_camera_far",
        "_fps",
        "_aspect_ratio",
        "_frame_height",
        "_pixel_height",
        "_window_pixel_height",
        "_start_frame_index",
        "_stop_frame_index",
        "_write_video",
        "_write_last_frame",
        "_preview",
        "_halt_on_last_frame"
    )

    def __init__(self) -> None:
        self._camera_altitude: float = 5.0
        self._camera_near: float = 0.1
        self._camera_far: float = 100.0
        self._fps: int = 30

        self._aspect_ratio: float = 16.0 / 9.0
        self._frame_height: float = 8.0
        self._pixel_height: float = 1080
        self._window_pixel_height: float = 540

        self._start_frame_index: int | None = None
        self._stop_frame_index: int | None = None
        self._write_video: bool = False
        self._write_last_frame: bool = False
        self._preview: bool = True
        self._halt_on_last_frame: bool = True

    # paths

    @classmethod
    def _ensure_directory_exists(
        cls,
        folder_path: pathlib.Path
    ) -> pathlib.Path:
        folder_path.mkdir(exist_ok=True)
        return folder_path

    @property
    def manim3_dir(self) -> pathlib.Path:
        return pathlib.Path(__file__).absolute().parent.parent

    @property
    def shaders_dir(self) -> pathlib.Path:
        return self.manim3_dir.joinpath("shaders")

    @property
    def tex_templates_path(self) -> pathlib.Path:
        return self.manim3_dir.joinpath("tex_templates.tml")

    @property
    def user_script_path(self) -> pathlib.Path:
        return pathlib.Path(sys.argv[0]).absolute()

    @property
    def output_dir(self) -> pathlib.Path:
        return self._ensure_directory_exists(self.user_script_path.parent.joinpath("manim3_files"))

    @property
    def tex_dir(self) -> pathlib.Path:
        return self._ensure_directory_exists(self.output_dir.joinpath("_tex"))

    @property
    def text_dir(self) -> pathlib.Path:
        return self._ensure_directory_exists(self.output_dir.joinpath("_text"))

    # camera

    @property
    def camera_altitude(self) -> float:
        return self._camera_altitude

    @property
    def camera_near(self) -> float:
        return self._camera_near

    @property
    def camera_far(self) -> float:
        return self._camera_far

    @property
    def fps(self) -> int:
        return self._fps

    @fps.setter
    def fps(
        self,
        fps: int
    ) -> None:
        self._fps = fps

    # size & resolution

    @property
    def aspect_ratio(self) -> float:
        return self._aspect_ratio

    @property
    def frame_height(self) -> float:
        return self._frame_height

    @property
    def frame_width(self) -> float:
        return self.aspect_ratio * self.frame_height

    @property
    def frame_size(self) -> tuple[float, float]:
        return (self.frame_width, self.frame_height)

    @property
    def pixel_height(self) -> float:
        return self._pixel_height

    @property
    def pixel_width(self) -> float:
        return self.aspect_ratio * self.pixel_height

    @property
    def pixel_size(self) -> tuple[int, int]:
        return (int(self.pixel_width), int(self.pixel_height))

    @property
    def pixel_per_unit(self) -> float:
        return self.pixel_height / self.frame_height

    @property
    def window_pixel_height(self) -> float:
        return self._window_pixel_height

    @property
    def window_pixel_width(self) -> float:
        return self.aspect_ratio * self.window_pixel_height

    @property
    def window_pixel_size(self) -> tuple[int, int]:
        return (int(self.window_pixel_width), int(self.window_pixel_height))

    @aspect_ratio.setter
    def aspect_ratio(
        self,
        aspect_ratio: float
    ) -> None:
        self._aspect_ratio = aspect_ratio

    @frame_size.setter
    def frame_size(
        self,
        frame_size: tuple[float, float]
    ) -> None:
        width, height = frame_size
        self._aspect_ratio = width / height
        self._frame_height = height

    @pixel_size.setter
    def pixel_size(
        self,
        pixel_size: tuple[float, float]
    ) -> None:
        width, height = pixel_size
        self._aspect_ratio = width / height
        self._pixel_height = height

    @window_pixel_size.setter
    def window_pixel_size(
        self,
        window_pixel_size: tuple[float, float]
    ) -> None:
        width, height = window_pixel_size
        self._aspect_ratio = width / height
        self._window_pixel_height = height

    # write mode

    @property
    def start_frame_index(self) -> int | None:
        return self._start_frame_index

    @property
    def start_time(self) -> float | None:
        return None if self.start_frame_index is None else self.start_frame_index / self.fps

    @property
    def stop_frame_index(self) -> int | None:
        return self._stop_frame_index

    @property
    def stop_time(self) -> float | None:
        return None if self.stop_frame_index is None else self.stop_frame_index / self.fps

    @property
    def time_span(self) -> tuple[float | None, float | None]:
        return (self.start_time, self.stop_time)

    @property
    def write_video(self) -> bool:
        return self._write_video

    @property
    def write_last_frame(self) -> bool:
        return self._write_last_frame

    @property
    def preview(self) -> bool:
        return self._preview

    @property
    def halt_on_last_frame(self) -> bool:
        return self._halt_on_last_frame

    @classmethod
    def _validate_frame_index_span(
        cls,
        start_frame_index: int | None,
        stop_frame_index: int | None
    ) -> None:
        assert (start_frame_index is None or start_frame_index >= 0) and (
            start_frame_index is None or stop_frame_index is None or start_frame_index <= stop_frame_index
        )

    @start_time.setter
    def start_time(
        self,
        start_time: float | None
    ) -> None:
        start_frame_index = None if start_time is None else int(start_time * self.fps)
        self._validate_frame_index_span(start_frame_index, self.stop_frame_index)
        self._start_frame_index = start_frame_index

    @stop_time.setter
    def stop_time(
        self,
        stop_time: float | None
    ) -> None:
        stop_frame_index = None if stop_time is None else int(stop_time * self.fps)
        self._validate_frame_index_span(self.start_frame_index, stop_frame_index)
        self._stop_frame_index = stop_frame_index

    @time_span.setter
    def time_span(
        self,
        time_span: tuple[float | None, float | None]
    ) -> None:
        start_time, stop_time = time_span
        start_frame_index = None if start_time is None else int(start_time * self.fps)
        stop_frame_index = None if stop_time is None else int(stop_time * self.fps)
        self._validate_frame_index_span(start_frame_index, stop_frame_index)
        self._start_frame_index = start_frame_index
        self._stop_frame_index = stop_frame_index

    @write_video.setter
    def write_video(
        self,
        write_video: bool
    ) -> None:
        self._write_video = write_video

    @write_last_frame.setter
    def write_last_frame(
        self,
        write_last_frame: bool
    ) -> None:
        self._write_last_frame = write_last_frame

    @preview.setter
    def preview(
        self,
        preview: bool
    ) -> None:
        self._preview = preview

    @halt_on_last_frame.setter
    def halt_on_last_frame(
        self,
        halt_on_last_frame: bool
    ) -> None:
        self._halt_on_last_frame = halt_on_last_frame


class ConfigSingleton(ABC):
    __slots__ = ()

    _INSTANCE: ClassVar[Config | None] = None

    def __new__(cls) -> Config:
        assert cls._INSTANCE is not None, "Config instance is not provided"
        return cls._INSTANCE

    @classmethod
    def set(
        cls,
        config: Config
    ) -> None:
        cls._INSTANCE = config
