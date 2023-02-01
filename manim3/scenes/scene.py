__all__ = ["Scene"]


import time

from ..config import Config
from ..custom_typing import Real
from ..scenes.child_scene import ChildScene
from ..utils.render_procedure import RenderProcedure


class Scene(ChildScene):
    def _render_scene(self) -> None:
        framebuffer = RenderProcedure._WINDOW_FRAMEBUFFER
        scene_config = self._scene_config
        red, green, blue = scene_config._background_color_
        alpha = scene_config._background_opacity_
        framebuffer.clear(red=red, green=green, blue=blue, alpha=alpha)
        self._render_with_passes(scene_config, framebuffer)

    def wait(self, t: Real):
        window = RenderProcedure._WINDOW
        #if window is None:
        #    return self  # TODO
        dt = 1.0 / Config.fps
        elapsed_time = 0.0
        timestamp = time.time()
        while not window.is_closing and elapsed_time < t:
            elapsed_time += dt
            delta_t = time.time() - timestamp
            if dt > delta_t:
                time.sleep(dt - delta_t)
            timestamp = time.time()
            window.clear()
            self._update_dt(dt)
            self._render_scene()
            window.swap_buffers()
        return self
