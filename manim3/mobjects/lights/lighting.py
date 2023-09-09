import numpy as np

from ...lazy.lazy import Lazy
from ...lazy.lazy_object import LazyObject
from ...rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from .ambient_light import AmbientLight
from .point_light import PointLight


class Lighting(LazyObject):
    __slots__ = ()

    def __init__(
        self,
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

    @Lazy.variable_collection(freeze=False)
    @staticmethod
    def _ambient_lights_() -> tuple[AmbientLight, ...]:
        return ()

    @Lazy.variable_collection(freeze=False)
    @staticmethod
    def _point_lights_() -> tuple[PointLight, ...]:
        return ()

    @Lazy.property()
    @staticmethod
    def _lighting_uniform_block_buffer_(
        ambient_lights: tuple[AmbientLight, ...],
        point_lights: tuple[PointLight, ...]
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
                "u_ambient_lights.color": np.fromiter((
                    ambient_light._color_._array_
                    for ambient_light in ambient_lights
                ), dtype=np.dtype((np.float64, (3,)))),
                "u_point_lights.position": np.fromiter((
                    point_light._position_
                    for point_light in point_lights
                ), dtype=np.dtype((np.float64, (3,)))),
                "u_point_lights.color": np.fromiter((
                    point_light._color_._array_
                    for point_light in point_lights
                ), dtype=np.dtype((np.float64, (3,))))
            }
        )
