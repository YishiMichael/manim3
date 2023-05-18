from dataclasses import dataclass
import pathlib
import sys
from typing import ClassVar

from colour import Color

from .custom_typing import ColorT


@dataclass(
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
        return pathlib.Path(__file__).absolute().parent

    @property
    def shaders_dir(self) -> pathlib.Path:
        return self.manim3_dir.joinpath("shaders")

    @property
    def tex_templates_path(self) -> pathlib.Path:
        return self.manim3_dir.joinpath("tex_templates.toml")

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
    kw_only=True,
    slots=True
)
class RenderingConfig:
    scene_name: str
    background_color: ColorT
    fps: int
    start_time: float
    stop_time: float | None
    write_video: bool
    write_last_frame: bool
    preview: bool


@dataclass(
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
    def frame_radii(self) -> tuple[float, float]:
        return (self.frame_width / 2.0, self.frame_height / 2.0)

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
    kw_only=True,
    slots=True
)
class CameraConfig:
    altitude: float
    near: float
    far: float


@dataclass(
    kw_only=True,
    slots=True
)
class TexConfig:
    use_mathjax: bool
    preamble: str | None
    template: str
    alignment: str | None
    environment: str | None
    base_color: ColorT
    font_size: float


@dataclass(
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
    base_color: ColorT
    line_spacing_height: float
    global_config: dict[str, str]
    language: str
    code_style: str


class Config:
    __slots__ = (
        "_path",
        "_rendering",
        "_size",
        "_camera",
        "_tex",
        "_text"
    )

    def __init__(self) -> None:
        super().__init__()
        self._path: PathConfig = PathConfig()
        self._rendering: RenderingConfig = RenderingConfig(
            scene_name=NotImplemented,  # Represents the class name.,
            background_color=Color("black"),
            fps=30,
            start_time=0.0,
            stop_time=None,
            write_video=False,
            write_last_frame=False,
            preview=True
        )
        self._size: SizeConfig = SizeConfig(
            aspect_ratio=16.0 / 9.0,
            frame_height=8.0,
            pixel_height=1080,
            window_pixel_height=540
        )
        self._camera: CameraConfig = CameraConfig(
            altitude=5.0,
            near=0.1,
            far=100.0
        )
        self._tex: TexConfig = TexConfig(
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
        self._text: TextConfig = TextConfig(
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

    @property
    def path(self) -> PathConfig:
        return self._path

    @property
    def rendering(self) -> RenderingConfig:
        return self._rendering

    @property
    def size(self) -> SizeConfig:
        return self._size

    @property
    def camera(self) -> CameraConfig:
        return self._camera

    @property
    def tex(self) -> TexConfig:
        return self._tex

    @property
    def text(self) -> TextConfig:
        return self._text


class ConfigSingleton:
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
