import weakref

from ..rendering.texture import TextureFactory
from .mesh_mobject import MeshMobject


class TexturedMobject(MeshMobject):
    __slots__ = ()

    def __init__(
        self,
        size: tuple[int, int] | None = None
    ) -> None:
        super().__init__()
        color_map_context_manager = TextureFactory.texture(size=size, components=3)
        self._color_map_ = color_map_context_manager.__enter__()
        weakref.finalize(
            self,
            lambda color_map_context_manager: color_map_context_manager.__exit__(None, None, None),
            color_map_context_manager
        )
