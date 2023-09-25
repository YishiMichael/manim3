import hashlib
import json
import pathlib
from abc import (
    ABC,
    abstractmethod
)
from contextlib import contextmanager
from typing import (
    Generic,
    Iterator,
    TypeVar
)

from ..utils.path_utils import PathUtils


_InputDataT = TypeVar("_InputDataT")
_JSONDataT = TypeVar("_JSONDataT")
_OutputDataT = TypeVar("_OutputDataT")


class MobjectIO(ABC, Generic[_InputDataT, _OutputDataT, _JSONDataT]):
    __slots__ = ()

    def __new__(cls):
        raise TypeError

    @classmethod
    def get(
        cls,
        input_data: _InputDataT
    ) -> _OutputDataT:
        # Notice that as we are using `str(input_data)` as key,
        # each item shall have an explicit string representation of data,
        # which shall not contain any information varying in each run, like addresses.
        hash_content = str(input_data)
        # Truncating at 16 bytes for cleanliness.
        hex_string = hashlib.sha256(hash_content.encode()).hexdigest()[:16]
        json_path = PathUtils.get_output_subdir(cls._dir_name).joinpath(f"{hex_string}.json")
        if not json_path.exists():
            with cls.display_during_execution():
                temp_path = PathUtils.get_output_subdir("_temp").joinpath(hex_string)
                output_data = cls.generate(input_data, temp_path)
                json_data = cls.dump_json(output_data)
                json_text = json.dumps(json_data, ensure_ascii=False)
                json_path.write_text(json_text, encoding="utf-8")
        json_text = json_path.read_text(encoding="utf-8")
        json_data = json.loads(json_text)
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

    @classmethod
    @contextmanager
    def display_during_execution(cls) -> Iterator[None]:  # TODO: needed?
        message = "Generating intermediate files..."
        try:
            print(message, end="\r")
            yield
        finally:
            print(" " * len(message), end="\r")
