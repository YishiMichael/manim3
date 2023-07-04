from typing import ContextManager

import moderngl

from ..rendering.texture import TextureFactory
from .mesh_mobject import MeshMobject


class TexturedMobject(MeshMobject):
    __slots__ = ("_color_map_context_manager",)

    def __init__(
        self,
        size: tuple[int, int] | None = None
    ) -> None:
        super().__init__()
        color_map_context_manager = TextureFactory.texture(size=size, components=3)
        self._color_map_context_manager: ContextManager[moderngl.Texture] = color_map_context_manager
        self._color_map_ = color_map_context_manager.__enter__()

    def __del__(self) -> None:
        self._color_map_context_manager.__exit__(None, None, None)  # TODO
