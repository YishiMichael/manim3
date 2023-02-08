#__all__ = [
#    "DepthTexture",
#    "Texture"
#]


#import atexit

#import moderngl

#from ..rendering.context import ContextSingleton
#from ..rendering.temporary_resource import TemporaryResource


#class Texture(TemporaryResource[tuple[tuple[int, int], int, int, str], moderngl.Texture]):
#    def __init__(
#        self,
#        *,
#        size: tuple[int, int],
#        components: int,
#        samples: int,
#        dtype: str
#    ):
#        super().__init__((size, components, samples, dtype))

#    @classmethod
#    def _new_instance(cls, parameters: tuple[tuple[int, int], int, int, str]) -> moderngl.Texture:
#        size, components, samples, dtype = parameters
#        texture = ContextSingleton().texture(
#            size=size,
#            components=components,
#            samples=samples,
#            dtype=dtype
#        )
#        atexit.register(lambda: texture.release())
#        return texture


#class DepthTexture(TemporaryResource[tuple[tuple[int, int], int], moderngl.Texture]):
#    def __init__(
#        self,
#        *,
#        size: tuple[int, int],
#        samples: int
#    ):
#        super().__init__((size, samples))

#    @classmethod
#    def _new_instance(cls, parameters: tuple[tuple[int, int], int]) -> moderngl.Texture:
#        size, samples = parameters
#        depth_texture = ContextSingleton().depth_texture(
#            size=size,
#            samples=samples
#        )
#        atexit.register(lambda: depth_texture.release())
#        return depth_texture
