
from ...lazy.lazy import Lazy
from ..mobject.mobject_attributes.mobject_attribute import MobjectAttribute
from .camera import Camera


class CameraAttribute(MobjectAttribute):
    __slots__ = ()

    def __init__(
        self,
        camera: Camera
    ) -> None:
        super().__init__()
        self._camera_ = camera

    @classmethod
    def _convert_input(
        cls,
        camera_input: Camera
    ) -> "CameraAttribute":
        return CameraAttribute(camera_input)

    @Lazy.variable(freeze=False)
    @staticmethod
    def _camera_() -> Camera:
        return Camera()
