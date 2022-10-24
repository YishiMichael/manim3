from __future__ import annotations

from cameras.orthographic_camera import OrthographicCamera
from lights.light_shadow import LightShadow


class DirectionalLightShadow(LightShadow):
	def __init__(self):
		super().__init__(OrthographicCamera(-5.0, 5.0, 5.0, -5.0, 0.5, 500.0))
