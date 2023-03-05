__all__ = ["OrthographicCamera"]


import numpy as np

from ..cameras.camera import Camera
from ..custom_typing import (
    Mat4T,
    Real
)
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..rendering.config import ConfigSingleton


class OrthographicCamera(Camera):
    __slots__ = ()

    def __init__(
        self,
        *,
        width: Real | None = None,
        height: Real | None = None,
        near: Real | None = None,
        far: Real | None = None
    ) -> None:
        super().__init__()
        if width is not None:
            self._width_ = width
        if height is not None:
            self._height_ = height
        if near is not None:
            self._near_ = near
        if far is not None:
            self._far_ = far

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _width_(cls) -> Real:
        return ConfigSingleton().frame_width

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _height_(cls) -> Real:
        return ConfigSingleton().frame_height

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _near_(cls) -> Real:
        return ConfigSingleton().camera_near

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _far_(cls) -> Real:
        return ConfigSingleton().camera_far

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _projection_matrix_(
        cls,
        width: Real,
        height: Real,
        near: Real,
        far: Real
    ) -> Mat4T:
        sx = 2.0 / width
        sy = 2.0 / height
        sz = -2.0 / (far - near)
        tz = -(far + near) / (far - near)
        return np.array((
            ( sx, 0.0, 0.0, 0.0),
            (0.0,  sy, 0.0, 0.0),
            (0.0, 0.0,  sz,  tz),
            (0.0, 0.0, 0.0, 1.0)
        ))
