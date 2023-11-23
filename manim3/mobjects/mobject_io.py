from __future__ import annotations


import hashlib
import json
import pathlib
from abc import (
    ABC,
    abstractmethod
)
from typing import (
    ClassVar,
    Never,
    Self,
    TypedDict
)

import attrs

from ..toplevel.toplevel import Toplevel


@attrs.frozen(kw_only=True)
class MobjectInput:
    pass


@attrs.frozen(kw_only=True)
class MobjectOutput:
    pass


class MobjectJSON(TypedDict):
    pass


class MobjectIO[MobjectInputT: MobjectInput, MobjectOutputT: MobjectOutput, MobjectJSONT: MobjectJSON](ABC):
    __slots__ = ()

    _dir_name: ClassVar[str]

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @classmethod
    def get(
        cls: type[Self],
        input_data: MobjectInputT
    ) -> MobjectOutputT:
        # Notice that as we are using `str(input_data)` as key,
        # each item shall have an explicit string representation of data,
        # which shall not contain any information varying in each run, like addresses.
        hash_content = str(input_data)
        # Truncating at 16 bytes for cleanliness.
        hex_string = hashlib.sha256(hash_content.encode()).hexdigest()[:16]
        json_path = cls._get_output_subdir(cls._dir_name).joinpath(f"{hex_string}.json")
        if json_path.exists():
            Toplevel._get_logger().log(f"Using cached intermediate files in {cls.__name__}.")
        else:
            Toplevel._get_logger().log(f"Generating intermediate files in {cls.__name__}...")
            temp_path = cls._get_output_subdir("_temp").joinpath(hex_string)
            output_data = cls.generate(input_data, temp_path)
            json_data = cls.dump_json(output_data)
            json_text = json.dumps(json_data, ensure_ascii=False)
            json_path.write_text(json_text, encoding="utf-8")
            Toplevel._get_logger().log(f"Intermediate files generation completed.")
        json_text = json_path.read_text(encoding="utf-8")
        json_data = json.loads(json_text)
        return cls.load_json(json_data)

    @classmethod
    def _get_output_subdir(
        cls: type[Self],
        dir_name: str
    ) -> pathlib.Path:
        subdir = Toplevel._get_config().output_dir.joinpath(dir_name)
        subdir.mkdir(exist_ok=True)
        return subdir

    @classmethod
    @abstractmethod
    def generate(
        cls: type[Self],
        input_data: MobjectInputT,
        temp_path: pathlib.Path
    ) -> MobjectOutputT:
        pass

    @classmethod
    @abstractmethod
    def dump_json(
        cls: type[Self],
        output_data: MobjectOutputT
    ) -> MobjectJSONT:
        pass

    @classmethod
    @abstractmethod
    def load_json(
        cls: type[Self],
        json_data: MobjectJSONT
    ) -> MobjectOutputT:
        pass
