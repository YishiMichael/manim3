from __future__ import annotations

from cameras.camera import Camera
from mobject import Mobject
from utils.arrays import Mat4, Vec2
from utils.texture import Texture


class LightShadow(Mobject):
    def __init__(self, camera: Camera):
        self.camera: Camera = camera
        self.bias: float = 0.0
        self.normalBias: float = 0.0
        self.radius: float = 1.0
        self.map_size: Vec2 = Vec2(512, 512)
        self.map: Texture | None = None
        self.matrix: Mat4 = Mat4()
