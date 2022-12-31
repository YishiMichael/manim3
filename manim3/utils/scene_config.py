__all__ = ["SceneConfig"]


from ..cameras.camera import Camera
from ..cameras.perspective_camera import PerspectiveCamera

from ..utils.lazy import LazyBase, lazy_property_initializer_writable


class SceneConfig(LazyBase):
    @lazy_property_initializer_writable
    @classmethod
    def _camera_(cls) -> Camera:
        return PerspectiveCamera()
