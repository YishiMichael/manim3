__all__ = ["Scene"]


import time

import moderngl
from moderngl_window.context.pyglet.window import Window as PygletWindow

from ..cameras.camera import Camera
from ..cameras.perspective_camera import PerspectiveCamera
from ..mobjects.mobject import Mobject
#from ..mobjects.scene_mobject import SceneMobject
from ..utils.context_singleton import ContextSingleton
from ..constants import PIXEL_HEIGHT, PIXEL_WIDTH
from ..custom_typing import *


class Scene(Mobject):
    def __init__(self, is_main: bool = False):
        super().__init__()
        self.mobject: Mobject = Mobject()
        self.camera: Camera = PerspectiveCamera()

        if is_main:
            window = PygletWindow(
                size=(PIXEL_WIDTH // 2, PIXEL_HEIGHT // 2),  # TODO
                fullscreen=False,
                resizable=True,
                gl_version=(3, 3),
                vsync=True,
                cursor=True
            )
            ctx = window.ctx
            ContextSingleton.set(ctx)
            framebuffer = ctx.detect_framebuffer()
        else:
            window = None
            #ctx = moderngl.create_context(standalone=True)
            #fbo = ctx.framebuffer(
            #    color_attachments=(ctx.texture(
            #        (PIXEL_WIDTH, PIXEL_HEIGHT),
            #        components=4,
            #        samples=0,
            #    ),),
            #    depth_attachment=ctx.depth_renderbuffer(
            #        (PIXEL_WIDTH, PIXEL_HEIGHT),
            #        samples=0
            #    )
            #)
            framebuffer = ContextSingleton().simple_framebuffer(
                size=(PIXEL_WIDTH, PIXEL_HEIGHT)
            )
            #fbo = ctx.detect_framebuffer()
            #fbo.use()

        #self.scene_mobject: SceneMobject = SceneMobject(
        #    camera=PerspectiveCamera(),
        #    frame=skia.Rect(-FRAME_X_RADIUS, -FRAME_Y_RADIUS, FRAME_X_RADIUS, FRAME_Y_RADIUS),
        #    resolution=(PIXEL_WIDTH, PIXEL_HEIGHT),
        #    window=PygletWindow(
        #        size=(PIXEL_WIDTH // 2, PIXEL_HEIGHT // 2),  # TODO
        #        fullscreen=False,
        #        resizable=True,
        #        gl_version=(3, 3),
        #        vsync=True,
        #        cursor=True
        #    )
        #)
        #if has_window:
        #    window = PygletWindow(
        #        size=(PIXEL_WIDTH // 2, PIXEL_HEIGHT // 2),  # TODO
        #        fullscreen=False,
        #        resizable=True,
        #        gl_version=(3, 3),
        #        vsync=True,
        #        cursor=True
        #    )
        #    ctx = window.ctx
        #    fbo = ctx.detect_framebuffer()
        #    fbo.use()
        #else:
        #    window = None
        #    ctx = moderngl.create_context(standalone=True)
        #    #fbo = ctx.framebuffer(
        #    #    color_attachments=(ctx.texture(
        #    #        (PIXEL_WIDTH, PIXEL_HEIGHT),
        #    #        components=4,
        #    #        samples=0,
        #    #    ),),
        #    #    depth_attachment=ctx.depth_renderbuffer(
        #    #        (PIXEL_WIDTH, PIXEL_HEIGHT),
        #    #        samples=0
        #    #    )
        #    #)
        #    fbo = ctx.simple_framebuffer(
        #        size=(PIXEL_WIDTH, PIXEL_HEIGHT)
        #    )
        #    #fbo = ctx.detect_framebuffer()
        #    fbo.use()

        #ContextSingleton.set(ctx)
        #fbo.use()
        self.window: PygletWindow | None = window
        #self.ctx: moderngl.Context = ctx
        #self.context_wrapper: ContextWrapper = ContextWrapper(ctx)
        self.framebuffer: moderngl.Framebuffer = framebuffer

        #ctx = moderngl.create_context(standalone=True)
        #self.context_wrapper: ContextWrapper = ContextWrapper(ctx)
        #ctx.enable(moderngl.DEPTH_TEST)
        #ctx.enable(moderngl.BLEND)
        #fbo = ctx.simple_framebuffer((960, 540))
        #fbo.use()
        #fbo.clear(0.0, 0.0, 0.0, 1.0)  # background color
        #self.ctx: moderngl.Context = ctx
        #self.fbo: moderngl.Framebuffer = fbo

    #def _init_framebuffer(self) -> moderngl.Framebuffer:
    #    return ContextSingleton().simple_framebuffer(
    #        size=(PIXEL_WIDTH, PIXEL_HEIGHT)
    #    )

    def add(self, *mobjects: Mobject):
        self.mobject.add(*mobjects)
        return self

    def remove(self, *mobjects: Mobject):
        self.mobject.remove(*mobjects)
        return self

    def _render(self, scene: "Scene", target_framebuffer: moderngl.Framebuffer) -> None:
        for mobject in self.mobject.get_descendants():
            mobject._render_full(scene, target_framebuffer)

    def _render_scene(self) -> None:
        self._render_full(self, self.framebuffer)

    def _update_scene_dt(self, dt: Real):
        for mobject in self.mobject.get_descendants():
            mobject._update_dt(dt)
        return self

    #def render(self, dt: float):
    #    for mobject in self.mobject.get_descendants():
    #        mobject.update_dt(dt)
    #        #mobject._camera_ = self.camera
    #        mobject.render(self)
    #    return self

    def wait(self, t: Real):
        window = self.window
        if window is None:  # TODO
            return self

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
            self._update_scene_dt(dt)
            self._render_scene()
            window.swap_buffers()
        return self
