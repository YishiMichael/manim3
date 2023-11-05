from __future__ import annotations


from typing import Self

import numpy as np

from ..constants.custom_typing import NP_3f8
from ..lazy.lazy import Lazy
from ..rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from .animatable.animatable import (
    Animatable,
    AnimatableMeta
)
from .lights.ambient_light import AmbientLight
from .lights.point_light import PointLight


class Lighting(Animatable):
    __slots__ = ()

    def __init__(
        self: Self,
        *lights: AmbientLight | PointLight
    ) -> None:
        super().__init__()
        ambient_lights: list[AmbientLight] = []
        point_lights: list[PointLight] = []
        for light in lights:
            if isinstance(light, AmbientLight):
                ambient_lights.append(light)
            else:
                point_lights.append(light)
        self._ambient_lights_ = tuple(ambient_lights)
        self._point_lights_ = tuple(point_lights)

    @AnimatableMeta.register_descriptor()
    @Lazy.volatile(plural=True)
    @staticmethod
    def _ambient_lights_() -> tuple[AmbientLight, ...]:
        return ()

    @AnimatableMeta.register_descriptor()
    @Lazy.volatile(plural=True)
    @staticmethod
    def _point_lights_() -> tuple[PointLight, ...]:
        return ()

    @Lazy.property()
    @staticmethod
    def _lighting_uniform_block_buffer_(
        ambient_lights__color__array: tuple[NP_3f8, ...],
        point_lights__color__array: tuple[NP_3f8, ...],
        point_lights__position: tuple[NP_3f8, ...]
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_lighting",
            fields=[
                "AmbientLight u_ambient_lights[NUM_U_AMBIENT_LIGHTS]",
                "PointLight u_point_lights[NUM_U_POINT_LIGHTS]"
            ],
            child_structs={
                "AmbientLight": [
                    "vec3 color"
                ],
                "PointLight": [
                    "vec3 position",
                    "vec3 color"
                ]
            },
            array_lens={
                "NUM_U_AMBIENT_LIGHTS": len(ambient_lights__color__array),
                "NUM_U_POINT_LIGHTS": len(point_lights__color__array)
            },
            data={
                "u_ambient_lights.color": np.fromiter(
                    ambient_lights__color__array,
                    dtype=np.dtype((np.float64, (3,)))
                ),
                "u_point_lights.position": np.fromiter(
                    point_lights__position,
                    dtype=np.dtype((np.float64, (3,)))
                ),
                "u_point_lights.color": np.fromiter(
                    point_lights__color__array,
                    dtype=np.dtype((np.float64, (3,)))
                )
            }
        )
