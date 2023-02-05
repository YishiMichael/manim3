__all__ = [
    "RenderPass",
    #"RenderPassSingleton"
]


from abc import abstractmethod

import moderngl

from ..utils.lazy import LazyBase


class RenderPass:
    @abstractmethod
    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> moderngl.Texture:
        pass


#class RenderPassSingleton(LazyBase):
#    _INSTANCE: "RenderPassSingleton | None"
#
#    def __init_subclass__(cls) -> None:
#        super().__init_subclass__()
#        cls._INSTANCE = None
#
#    def __new__(cls):
#        if (instance := cls._INSTANCE) is not None:
#            assert isinstance(instance, cls)
#            return instance
#        instance = super().__new__(cls)
#        cls._INSTANCE = instance
#        return instance
