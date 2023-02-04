__all__ = [
    "RenderPass",
    "RenderPassSingleton"
]


from abc import abstractmethod

import moderngl

from ..utils.lazy import LazyBase


class RenderPass:
    @abstractmethod
    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> moderngl.Texture:
        pass


class RenderPassSingleton(LazyBase):
    _INSTANCES: "dict[type[RenderPassSingleton], RenderPassSingleton]" = {}

    def __new__(cls):
        if (instance := cls._INSTANCES.get(cls)) is not None:
            assert isinstance(instance, cls)
            return instance
        instance = super().__new__(cls)
        cls._INSTANCES[cls] = instance
        return instance
