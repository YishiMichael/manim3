__all__ = [
    "Config",
    "ConfigSingleton"
]


from abc import ABC
from dataclasses import dataclass
import pathlib
import sys
from typing import ClassVar

from colour import Color

from ..custom_typing import ColorType


@dataclass(
    order=True,
    kw_only=True,
    slots=True
)
class PathConfig:
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

@dataclass(
    order=True,
    kw_only=True,
    slots=True
)
class RenderingConfig:
    fps: int
    start_frame_index: int | None
    stop_frame_index: int | None
    write_video: bool
    write_last_frame: bool
    preview: bool

    @property
    def start_time(self) -> float | None:
        return None if self.start_frame_index is None else self.start_frame_index / self.fps

    @property
    def stop_time(self) -> float | None:
        return None if self.stop_frame_index is None else self.stop_frame_index / self.fps

    @property
    def time_span(self) -> tuple[float | None, float | None]:
        return (self.start_time, self.stop_time)

    @start_time.setter
    def start_time(
        self,
        start_time: float | None
    ) -> None:
        start_frame_index = None if start_time is None else int(start_time * self.fps)
        self._validate_frame_index_span(start_frame_index, self.stop_frame_index)
        self.start_frame_index = start_frame_index

    @stop_time.setter
    def stop_time(
        self,
        stop_time: float | None
    ) -> None:
        stop_frame_index = None if stop_time is None else int(stop_time * self.fps)
        self._validate_frame_index_span(self.start_frame_index, stop_frame_index)
        self.stop_frame_index = stop_frame_index

    @time_span.setter
    def time_span(
        self,
        time_span: tuple[float | None, float | None]
    ) -> None:
        start_time, stop_time = time_span
        start_frame_index = None if start_time is None else int(start_time * self.fps)
        stop_frame_index = None if stop_time is None else int(stop_time * self.fps)
        self._validate_frame_index_span(start_frame_index, stop_frame_index)
        self.start_frame_index = start_frame_index
        self.stop_frame_index = stop_frame_index

    @classmethod
    def _validate_frame_index_span(
        cls,
        start_frame_index: int | None,
        stop_frame_index: int | None
    ) -> None:
        assert (start_frame_index is None or start_frame_index >= 0) and (
            start_frame_index is None or stop_frame_index is None or start_frame_index <= stop_frame_index
        )


@dataclass(
    order=True,
    kw_only=True,
    slots=True
)
class SizeConfig:
    aspect_ratio: float
    frame_height: float
    pixel_height: float
    window_pixel_height: float

    @property
    def frame_width(self) -> float:
        return self.aspect_ratio * self.frame_height

    @property
    def frame_size(self) -> tuple[float, float]:
        return (self.frame_width, self.frame_height)

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
    def window_pixel_width(self) -> float:
        return self.aspect_ratio * self.window_pixel_height

    @property
    def window_pixel_size(self) -> tuple[int, int]:
        return (int(self.window_pixel_width), int(self.window_pixel_height))

    @frame_size.setter
    def frame_size(
        self,
        frame_size: tuple[float, float]
    ) -> None:
        width, height = frame_size
        self.aspect_ratio = width / height
        self.frame_height = height

    @pixel_size.setter
    def pixel_size(
        self,
        pixel_size: tuple[float, float]
    ) -> None:
        width, height = pixel_size
        self.aspect_ratio = width / height
        self.pixel_height = height

    @window_pixel_size.setter
    def window_pixel_size(
        self,
        window_pixel_size: tuple[float, float]
    ) -> None:
        width, height = window_pixel_size
        self.aspect_ratio = width / height
        self.window_pixel_height = height


@dataclass(
    order=True,
    kw_only=True,
    slots=True
)
class CameraConfig:
    altitude: float
    near: float
    far: float


@dataclass(
    order=True,
    kw_only=True,
    slots=True
)
class TexConfig:
    use_mathjax: bool
    preamble: str
    template: str
    alignment: str | None
    environment: str | None
    base_color: ColorType
    font_size: float


@dataclass(
    order=True,
    kw_only=True,
    slots=True
)
class TextConfig:
    justify: bool
    indent: float
    alignment: str
    line_width: float | None
    font_size: float
    font: str
    slant: str
    weight: str
    base_color: ColorType
    line_spacing_height: float
    global_config: dict[str, str]
    language: str
    code_style: str


@dataclass(
    order=True,
    kw_only=True,
    slots=True
)
class Config:
    path: PathConfig = PathConfig()
    rendering: RenderingConfig = RenderingConfig(
        fps=30,
        start_frame_index=None,
        stop_frame_index=None,
        write_video=False,
        write_last_frame=False,
        preview=True
    )
    size: SizeConfig = SizeConfig(
        aspect_ratio=16.0 / 9.0,
        frame_height=8.0,
        pixel_height=1080,
        window_pixel_height=540
    )
    camera: CameraConfig = CameraConfig(
        altitude=5.0,
        near=0.1,
        far=100.0
    )
    tex: TexConfig = TexConfig(
        use_mathjax=False,
        preamble="\n".join((
            "\\documentclass[preview]{standalone}",
            "\\usepackage{amsmath}",
            "\\usepackage{amssymb}",
            "\\usepackage{xcolor}"  # Required for labelling.
        )),
        template="ctex",
        alignment="\\centering",
        environment="align*",
        base_color=Color("white"),
        font_size=48
    )
    text: TextConfig = TextConfig(
        justify=False,
        indent=0.0,
        alignment="LEFT",
        line_width=None,
        font_size=48,
        font="Consolas",
        slant="NORMAL",
        weight="NORMAL",
        base_color=Color("white"),
        line_spacing_height=0.0,
        global_config={},
        language="python",
        # Visit `https://pygments.org/demo/` to have a preview of more styles.
        code_style="monokai"
    )


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
