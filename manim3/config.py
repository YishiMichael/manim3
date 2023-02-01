__all__ = ["Config"]


import os
import sys
from typing import ClassVar

from .custom_typing import Real


class Config:
    def __new__(cls):
        raise NotImplementedError

    def __init_subclass__(cls) -> None:
        raise NotImplementedError

    # size & resolution

    _ASPECT_RATIO: ClassVar[Real] = 16.0 / 9.0
    _FRAME_HEIGHT: ClassVar[Real] = 8.0
    _PIXEL_HEIGHT: ClassVar[Real] = 1080
    _WINDOW_PIXEL_HEIGHT: ClassVar[Real] = 540

    @classmethod
    @property
    def aspect_ratio(cls) -> Real:
        return cls._ASPECT_RATIO

    @classmethod
    @property
    def frame_height(cls) -> Real:
        return cls._FRAME_HEIGHT

    @classmethod
    @property
    def frame_width(cls) -> Real:
        return cls.aspect_ratio * cls.frame_height

    @classmethod
    @property
    def frame_size(cls) -> tuple[Real, Real]:
        return (cls.frame_width, cls.frame_height)

    @classmethod
    @property
    def pixel_height(cls) -> Real:
        return cls._PIXEL_HEIGHT

    @classmethod
    @property
    def pixel_width(cls) -> Real:
        return cls.aspect_ratio * cls.pixel_height

    @classmethod
    @property
    def pixel_size(cls) -> tuple[int, int]:
        return (int(cls.pixel_width), int(cls.pixel_height))

    @classmethod
    @property
    def pixel_per_unit(cls) -> Real:
        return cls.pixel_height / cls.frame_height

    @classmethod
    @property
    def window_pixel_height(cls) -> Real:
        return cls._WINDOW_PIXEL_HEIGHT

    @classmethod
    @property
    def window_pixel_width(cls) -> Real:
        return cls.aspect_ratio * cls.window_pixel_height

    @classmethod
    @property
    def window_pixel_size(cls) -> tuple[int, int]:
        return (int(cls.window_pixel_width), int(cls.window_pixel_height))

    @classmethod
    def set_aspect_ratio(cls, aspect_ratio: Real):
        cls._ASPECT_RATIO = aspect_ratio
        return cls

    @classmethod
    def set_frame_size(cls, width: Real, height: Real):
        cls._ASPECT_RATIO = width / height
        cls._FRAME_HEIGHT = height
        return cls

    @classmethod
    def set_pixel_size(cls, width: Real, height: Real):
        cls._ASPECT_RATIO = width / height
        cls._PIXEL_HEIGHT = height
        return cls

    @classmethod
    def set_window_pixel_size(cls, width: Real, height: Real):
        cls._ASPECT_RATIO = width / height
        cls._WINDOW_PIXEL_HEIGHT = height
        return cls

    # camera

    _CAMERA_ALTITUDE: ClassVar[Real] = 5.0
    _CAMERA_NEAR: ClassVar[Real] = 0.1
    _CAMERA_FAR: ClassVar[Real] = 100.0
    _FPS: ClassVar[Real] = 30.0

    @classmethod
    @property
    def camera_altitude(cls) -> Real:
        return cls._CAMERA_ALTITUDE

    @classmethod
    @property
    def camera_near(cls) -> Real:
        return cls._CAMERA_NEAR

    @classmethod
    @property
    def camera_far(cls) -> Real:
        return cls._CAMERA_FAR

    @classmethod
    @property
    def fps(cls) -> Real:
        return cls._FPS

    @classmethod
    def set_fps(cls, fps: Real):
        cls._FPS = fps
        return cls

    # paths

    @classmethod
    def _ensure_directory_exists(cls, folder_path: str) -> str:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        return folder_path

    @classmethod
    @property
    def manim3_dir(cls) -> str:
        return os.path.dirname(__file__)

    @classmethod
    @property
    def shaders_dir(cls) -> str:
        return os.path.join(cls.manim3_dir, "shaders")

    @classmethod
    @property
    def tex_templates_path(cls) -> str:
        return os.path.join(cls.manim3_dir, "tex_templates.tml")

    @classmethod
    @property
    def user_script_path(cls) -> str:
        return os.path.abspath(sys.argv[0])

    @classmethod
    @property
    def output_dir(cls) -> str:
        return cls._ensure_directory_exists(os.path.join(os.path.dirname(cls.user_script_path), "manim3_files"))

    @classmethod
    @property
    def tex_dir(cls) -> str:
        return cls._ensure_directory_exists(os.path.join(cls.output_dir, "_tex"))

    @classmethod
    @property
    def text_dir(cls) -> str:
        return cls._ensure_directory_exists(os.path.join(cls.output_dir, "_text"))
