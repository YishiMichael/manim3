__all__ = ["RenderPass"]


from abc import abstractmethod

import moderngl
import numpy as np

from ..rendering.glsl_variables import (
    AttributesBuffer,
    IndexBuffer
)
from ..utils.lazy import (
    LazyBase,
    lazy_basedata
)


class RenderPass(LazyBase):
    __slots__ = ()

    @lazy_basedata
    @staticmethod
    def _attributes_() -> AttributesBuffer:
        return AttributesBuffer(
            fields=[
                "vec3 in_position",
                "vec2 in_uv"
            ],
            num_vertex=4,
            data={
                "in_position": np.array((
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                )),
                "in_uv": np.array((
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ))
            }
        )

    @lazy_basedata
    @staticmethod
    def _index_buffer_() -> IndexBuffer:
        return IndexBuffer(
            data=np.array((
                0, 1, 2, 3
            ))
        )

    @abstractmethod
    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> moderngl.Texture:
        pass
