import hashlib
import json
from abc import (
    ABC,
    abstractmethod
)
from contextlib import contextmanager
import pathlib
from typing import (
    Generic,
    Iterator,
    TypeVar
)

from ...utils.path_utils import PathUtils


_InputDataT = TypeVar("_InputDataT")
_JSONDataT = TypeVar("_JSONDataT")
_OutputDataT = TypeVar("_OutputDataT")


class MobjectIO(ABC, Generic[_InputDataT, _OutputDataT, _JSONDataT]):
    __slots__ = ()

    #_dir_name: ClassVar[str]

    def __new__(cls):
        raise TypeError

    @classmethod
    def get(
        cls,
        input_data: _InputDataT
    ) -> _OutputDataT:
        hash_content = str(input_data)
        # Truncating at 16 bytes for cleanliness.
        hex_string = hashlib.sha256(hash_content.encode()).hexdigest()[:16]
        json_path = PathUtils.get_output_subdir(cls._dir_name).joinpath(f"{hex_string}.json")
        #svg_path = cls.get_hash_path(
        #    hash_content=hash_content,
        #    dir_name=cls._dir_name,
        #    suffix=".svg"
        #)
        if not json_path.exists():
            with cls.display_during_execution():
                temp_path = PathUtils.get_output_subdir("_temp").joinpath(f"{hex_string}.json")
                output_data = cls.generate(input_data, temp_path)
                json_data = cls.dump_json(output_data)
                with open(json_path, "w", encoding="utf-8") as json_file:
                    json.dump(json_data, json_file, ensure_ascii=False)
        with open(json_path, encoding="utf-8") as json_file:
            json_data = json.load(json_file)
        return cls.load_json(json_data)

    @classmethod
    @property
    @abstractmethod
    def _dir_name(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def generate(
        cls,
        input_data: _InputDataT,
        temp_path: pathlib.Path
    ) -> _OutputDataT:
        pass

    @classmethod
    @abstractmethod
    def dump_json(
        cls,
        output_data: _OutputDataT
    ) -> _JSONDataT:
        pass

    @classmethod
    @abstractmethod
    def load_json(
        cls,
        json_data: _JSONDataT
    ) -> _OutputDataT:
        pass

    #@classmethod
    #def get_hash_path(
    #    cls,
    #    hash_content: str,
    #    dir_name: str,
    #    suffix: str
    #) -> pathlib.Path:
    #    # Truncating at 16 bytes for cleanliness.
    #    hex_string = hashlib.sha256(hash_content.encode()).hexdigest()[:16]
    #    svg_dir = PathUtils.get_output_subdir(dir_name)
    #    return svg_dir.joinpath(f"{hex_string}{suffix}")

    @classmethod
    @contextmanager
    def display_during_execution(cls) -> Iterator[None]:  # TODO: needed?
        message = "Generating intermediate files..."
        try:
            print(message, end="\r")
            yield
        finally:
            print(" " * len(message), end="\r")
