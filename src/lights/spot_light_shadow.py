from __future__ import annotations

from cameras.perspective_camera import PerspectiveCamera
from lights.light_shadow import LightShadow


class SpotLightShadow(LightShadow):
	def __init__(self):
		super().__init__(PerspectiveCamera(50.0, 1.0, 0.5, 500.0))
		# self.focus: float = 1.0
