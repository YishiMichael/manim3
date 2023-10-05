from __future__ import annotations


import pathlib
import sys
from typing import (
    Never,
    Self
)


class PathUtils:
    __slots__ = ()

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @classmethod
    def _ensure_directory_exists(
        cls: type[Self],
        folder_path: pathlib.Path
    ) -> pathlib.Path:
        folder_path.mkdir(exist_ok=True)
        return folder_path

    @classmethod
    @property
    def src_dir(
        cls: type[Self]
    ) -> pathlib.Path:
        return pathlib.Path(__file__).absolute().parent.parent

    @classmethod
    @property
    def shaders_dir(
        cls: type[Self]
    ) -> pathlib.Path:
        return cls.src_dir.joinpath("shaders")

    @classmethod
    @property
    def plugins_dir(
        cls: type[Self]
    ) -> pathlib.Path:
        return cls.src_dir.joinpath("plugins")

    @classmethod
    @property
    def user_script_path(
        cls: type[Self]
    ) -> pathlib.Path:
        return pathlib.Path(sys.argv[0]).absolute()

    @classmethod
    @property
    def output_dir(
        cls: type[Self]
    ) -> pathlib.Path:
        return cls._ensure_directory_exists(cls.user_script_path.parent.joinpath("manim3_output"))

    @classmethod
    def get_output_subdir(
        cls: type[Self],
        dir_name: str
    ) -> pathlib.Path:
        return cls._ensure_directory_exists(cls.output_dir.joinpath(dir_name))
