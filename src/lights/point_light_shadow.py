from __future__ import annotations

from cameras.perspective_camera import PerspectiveCamera
from lights.light_shadow import LightShadow


class PointLightShadow(LightShadow):
	def __init__(self):
		super().__init__(PerspectiveCamera(90.0, 1.0, 0.5, 500.0))
