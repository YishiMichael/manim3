__all__ = ["Scene"]


import time

from moderngl_window.context.pyglet.window import Window as PygletWindow

from ..constants import (
    PIXEL_HEIGHT,
    PIXEL_WIDTH
)
from ..custom_typing import Real
from ..mobjects.mobject import Mobject
from ..utils.context_singleton import ContextSingleton
from ..utils.lazy import lazy_property_initializer
from ..utils.renderable import (
    Framebuffer,
    IntermediateFramebuffer
)
from ..utils.scene_config import SceneConfig


class Scene(Mobject):
    def __init__(self, *, main: bool = False):
        if main:
            window = PygletWindow(
                size=(PIXEL_WIDTH // 2, PIXEL_HEIGHT // 2),  # TODO
                fullscreen=False,
                resizable=True,
                gl_version=(4, 3),
                vsync=True,
                cursor=True
            )
            ctx = window.ctx
            #ctx.gc_mode = "auto"
            ContextSingleton.set(ctx)
            framebuffer = Framebuffer(
                ContextSingleton().detect_framebuffer()
            )
        else:
            window = None
            framebuffer = IntermediateFramebuffer(
                color_attachments=[
                    ContextSingleton().texture(
                        size=(PIXEL_WIDTH, PIXEL_HEIGHT),
                        components=4
                    )
                ],
                depth_attachment=ContextSingleton().depth_texture(
                    size=(PIXEL_WIDTH, PIXEL_HEIGHT)
                )
            )

        self._window: PygletWindow | None = window
        self._framebuffer: Framebuffer = framebuffer
        super().__init__()

    @lazy_property_initializer
    @staticmethod
    def _scene_config_() -> SceneConfig:
        return SceneConfig()

    def _render(self, scene_config: SceneConfig, target_framebuffer: Framebuffer) -> None:
        for mobject in self.get_descendants_excluding_self():
            mobject._render_full(scene_config, target_framebuffer)

    def _render_scene(self) -> None:
        framebuffer = self._framebuffer
        #framebuffer._framebuffer.clear()  # TODO: needed?
        self._render_full(self._scene_config_, framebuffer)

    def _update_dt(self, dt: Real):
        for mobject in self.get_descendants_excluding_self():
            mobject._update_dt(dt)
        return self

    def wait(self, t: Real):
        window = self._window
        if window is None:
            return self  # TODO

        FPS = 30.0
        dt = 1.0 / FPS
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
