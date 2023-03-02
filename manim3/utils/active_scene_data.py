__all__ = ["ActiveSceneDataSingleton"]


from dataclasses import dataclass
import subprocess as sp
from typing import ClassVar

import moderngl


@dataclass(
    slots=True,
    kw_only=True,
    frozen=True
)
class ActiveSceneData:
    color_texture: moderngl.Texture
    framebuffer: moderngl.Framebuffer
    writing_process: sp.Popen | None


class ActiveSceneDataSingleton:
    _INSTANCE: ClassVar[ActiveSceneData | None] = None

    def __new__(cls) -> ActiveSceneData:
        assert cls._INSTANCE is not None, "ActiveSceneData instance is not provided"
        return cls._INSTANCE

    @classmethod
    def set(
        cls,
        color_texture: moderngl.Texture,
        framebuffer: moderngl.Framebuffer,
        writing_process: sp.Popen | None
    ) -> None:
        cls._INSTANCE = ActiveSceneData(
            color_texture=color_texture,
            framebuffer=framebuffer,
            writing_process=writing_process
        )
