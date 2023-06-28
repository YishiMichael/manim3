import numpy as np

from ...lazy.lazy import (
    Lazy,
    LazyObject
)
from ...rendering.gl_buffer import UniformBlockBuffer
from .ambient_light import AmbientLight
from .point_light import PointLight


class Lighting(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        *lights: AmbientLight | PointLight
    ) -> None:
        super().__init__()
        for light in lights:
            if isinstance(light, AmbientLight):
                self._ambient_lights_.append(light)
            else:
                self._point_lights_.append(light)

    @Lazy.variable_collection
    @classmethod
    def _ambient_lights_(cls) -> list[AmbientLight]:
        return []

    @Lazy.variable_collection
    @classmethod
    def _point_lights_(cls) -> list[PointLight]:
        return []

    @Lazy.property
    @classmethod
    def _lighting_uniform_block_buffer_(
        cls,
        ambient_lights: list[AmbientLight],
        point_lights: list[PointLight]
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
                "NUM_U_AMBIENT_LIGHTS": len(ambient_lights),
                "NUM_U_POINT_LIGHTS": len(point_lights)
            },
            data={
                "u_ambient_lights.color": np.array([
                    ambient_light._color_
                    for ambient_light in ambient_lights
                ]),
                "u_point_lights.position": np.array([
                    point_light._position_
                    for point_light in point_lights
                ]),
                "u_point_lights.color": np.array([
                    point_light._color_
                    for point_light in point_lights
                ])
            }
        )
