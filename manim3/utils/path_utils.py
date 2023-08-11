import pathlib
import sys


class PathUtils:
    __slots__ = ()

    def __new__(cls):
        raise TypeError

    @classmethod
    def _ensure_directory_exists(
        cls,
        folder_path: pathlib.Path
    ) -> pathlib.Path:
        folder_path.mkdir(exist_ok=True)
        return folder_path

    @classmethod
    @property
    def src_dir(cls) -> pathlib.Path:
        return pathlib.Path(__file__).absolute().parent.parent

    @classmethod
    @property
    def shaders_dir(cls) -> pathlib.Path:
        return cls.src_dir.joinpath("shaders")

    @classmethod
    @property
    def plugins_dir(cls) -> pathlib.Path:
        return cls.src_dir.joinpath("plugins")

    @classmethod
    @property
    def user_script_path(cls) -> pathlib.Path:
        return pathlib.Path(sys.argv[0]).absolute()

    @classmethod
    @property
    def output_dir(cls) -> pathlib.Path:
        return cls._ensure_directory_exists(cls.user_script_path.parent.joinpath("manim3_output"))

    @classmethod
    def get_output_subdir(
        cls,
        dir_name: str
    ) -> pathlib.Path:
        return cls._ensure_directory_exists(cls.output_dir.joinpath(dir_name))
