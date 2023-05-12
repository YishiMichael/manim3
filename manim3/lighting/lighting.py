import numpy as np

from ..lazy.lazy import (
    Lazy,
    LazyObject
)
from ..lighting.ambient_light import AmbientLight
from ..lighting.point_light import PointLight
from ..rendering.gl_buffer import UniformBlockBuffer


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
        _ambient_lights_: list[AmbientLight],
        _point_lights_: list[PointLight]
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
                "NUM_U_AMBIENT_LIGHTS": len(_ambient_lights_),
                "NUM_U_POINT_LIGHTS": len(_point_lights_)
            },
            data={
                "u_ambient_lights.color": np.array([
                    ambient_light._color_vec4_.value
                    for ambient_light in _ambient_lights_
                ]),
                "u_point_lights.position": np.array([
                    point_light._position_.value
                    for point_light in _point_lights_
                ]),
                "u_point_lights.color": np.array([
                    point_light._color_vec4_.value
                    for point_light in _point_lights_
                ])
            }
        )
