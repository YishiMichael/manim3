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
                    "vec4 color"
                ],
                "PointLight": [
                    "vec3 position",
                    "vec4 color"
                ]
            },
            array_lens={
                "NUM_U_AMBIENT_LIGHTS": len(ambient_lights),
                "NUM_U_POINT_LIGHTS": len(point_lights)
            },
            data={
                "u_ambient_lights.color": np.array([
                    np.append(ambient_light._color_, ambient_light._opacity_)
                    for ambient_light in ambient_lights
                ]),
                "u_point_lights.position": np.array([
                    point_light._position_
                    for point_light in point_lights
                ]),
                "u_point_lights.color": np.array([
                    np.append(point_light._color_, point_light._opacity_)
                    for point_light in point_lights
                ])
            }
        )
