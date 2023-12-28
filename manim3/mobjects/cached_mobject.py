from __future__ import annotations


import hashlib
import json
import pathlib
from abc import abstractmethod
from typing import (
    Self,
    TypedDict
)

import attrs
import numpy as np

from ..animatables.shape import Shape
from ..toplevel.toplevel import Toplevel
from .shape_mobjects.shape_mobject import ShapeMobject


class ShapeMobjectJSON(TypedDict):
    coordinates: tuple[float, ...]  # flattened
    counts: tuple[int, ...]
    color: str   # hex rgba


@attrs.frozen(kw_only=True)
class CachedMobjectInputs:
    pass


class CachedMobject[CachedMobjectInputsT: CachedMobjectInputs](ShapeMobject):
    __slots__ = ("_shape_mobjects",)

    def __init__(
        self: Self,
        inputs: CachedMobjectInputsT
    ) -> None:
        super().__init__()
        shape_mobjects = type(self)._get_shape_mobjects(inputs)
        self._shape_mobjects: tuple[ShapeMobject, ...] = shape_mobjects
        self.add(*shape_mobjects)

    @classmethod
    def _get_shape_mobjects(
        cls: type[Self],
        inputs: CachedMobjectInputsT
    ) -> tuple[ShapeMobject, ...]:
        # Notice that as we are using string as key,
        # each item shall have an explicit string representation of data,
        # which shall not contain any information varying in each run, like addresses.
        hash_content = f"{inputs}"
        # Truncating at 16 bytes for cleanliness.
        hex_string = hashlib.sha256(hash_content.encode()).hexdigest()[:16]
        cache_storager = Toplevel._get_renderer()._cache_storager
        json_path = cache_storager.get_cache_path(f"{hex_string}.json")

        if not json_path.exists():
            temp_path = cache_storager.get_temp_path(hex_string)
            shape_mobjects = cls._generate_shape_mobjects(inputs, temp_path)
            shape_mobjects_json_tuple = tuple(
                cls._shape_mobject_to_json(shape_mobject)
                for shape_mobject in shape_mobjects
            )
            json_path.write_text(json.dumps(shape_mobjects_json_tuple, ensure_ascii=False), encoding="utf-8")

        shape_mobjects_json_tuple: tuple[ShapeMobjectJSON, ...] = json.loads(json_path.read_text(encoding="utf-8"))
        return tuple(
            cls._json_to_shape_mobject(shape_mobject_json)
            for shape_mobject_json in shape_mobjects_json_tuple
        )

    @classmethod
    @abstractmethod
    def _generate_shape_mobjects(
        cls: type[Self],
        inputs: CachedMobjectInputsT,
        temp_path: pathlib.Path
    ) -> tuple[ShapeMobject, ...]:
        pass

    @classmethod
    def _shape_mobject_to_json(
        cls: type[Self],
        shape_mobject: ShapeMobject
    ) -> ShapeMobjectJSON:
        shape = shape_mobject._shape_
        rgba_components = np.append(shape_mobject._color_._array_, shape_mobject._opacity_._array_)
        return ShapeMobjectJSON(
            coordinates=tuple(round(float(value), 6) for value in shape._coordinates_.flatten()),
            counts=tuple(int(value) for value in shape._counts_),
            color=f"#{"".join(f"{component:02X}" for component in (255.0 * rgba_components).astype(np.uint8))}"
        )

    @classmethod
    def _json_to_shape_mobject(
        cls: type[Self],
        shape_mobject_json: ShapeMobjectJSON
    ) -> ShapeMobject:
        coordinates = shape_mobject_json["coordinates"]
        counts = shape_mobject_json["counts"]
        color = shape_mobject_json["color"]
        rgba_components = np.fromiter((int(color[start:start + 2], 16) for start in range(1, 9, 2)), dtype=np.float64) / 255.0
        return ShapeMobject(Shape(
            coordinates=np.fromiter(coordinates, dtype=np.float64).reshape(-1, 2),
            counts=np.fromiter(counts, dtype=np.int32)
        )).set(
            color=rgba_components[:3],
            opacity=float(rgba_components[3])
        )
