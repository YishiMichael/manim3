from dataclasses import dataclass

from colour import Color

from ..constants.custom_typing import (
    AlignmentT,
    ColorT
)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class Config:
    fps: int = 30
    write_video: bool = False
    write_last_frame: bool = False
    preview: bool = True
    gl_version: tuple[int, int] = (4, 3)

    aspect_ratio: float = 16.0 / 9.0
    frame_height: float = 8.0
    pixel_height: float = 1080
    window_pixel_height: float = 540

    camera_distance: float = 5.0
    camera_near: float = 0.1
    camera_far: float = 100.0

    background_color: ColorT = Color("black")
    mesh_specular_strength: float = 0.5
    mesh_shininess: float = 32.0
    graph_width: float = 0.05

    tex_color: ColorT = Color("white")
    tex_font_size: float = 30
    tex_alignment: AlignmentT = "center"
    tex_compiler: str = "xelatex"
    tex_preamble: str = "\n".join((
        "\\documentclass[preview]{standalone}",
        "\\usepackage[UTF8]{ctex}",
        "\\usepackage{amsmath}",
        "\\usepackage{amssymb}",
        "\\usepackage{xcolor}"  # Required for labelling.
    ))

    text_color: ColorT = Color("white")
    text_font_size: float = 30
    text_alignment: AlignmentT = "center"
    text_font: str = "Consolas"
    text_justify: bool = False
    text_indent: float = 0.0
    text_line_width: float = -1.0

    @property
    def gl_version_code(self) -> int:
        major_version, minor_version = self.gl_version
        return major_version * 100 + minor_version * 10

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
