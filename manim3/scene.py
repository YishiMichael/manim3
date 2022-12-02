__all__ = ["Scene"]


import time

import moderngl
from moderngl_window.context.pyglet.window import Window as PygletWindow

from .cameras.camera import Camera
from .cameras.perspective_camera import PerspectiveCamera
from .mobjects.mobject import Mobject
#from ..mobjects.scene_mobject import SceneMobject
from .utils.context import ContextSingleton
from .constants import PIXEL_HEIGHT, PIXEL_WIDTH
from .custom_typing import *


class Scene(Mobject):
    def __init__(self, has_window: bool = True):
        super().__init__()
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
        if has_window:
            window = PygletWindow(
                size=(PIXEL_WIDTH // 2, PIXEL_HEIGHT // 2),  # TODO
                fullscreen=False,
                resizable=True,
                gl_version=(3, 3),
                vsync=True,
                cursor=True
            )
            ctx = window.ctx
            fbo = ctx.detect_framebuffer()
            fbo.use()
        else:
            window = None
            ctx = moderngl.create_context(standalone=True)
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
            fbo = ctx.simple_framebuffer(
                size=(PIXEL_WIDTH, PIXEL_HEIGHT)
            )
            #fbo = ctx.detect_framebuffer()
            fbo.use()

        ContextSingleton.set(ctx)
        #fbo.use()
        self.window: PygletWindow | None = window
        #self.ctx: moderngl.Context = ctx
        #self.context_wrapper: ContextWrapper = ContextWrapper(ctx)
        self.fbo: moderngl.Framebuffer = fbo
        self.camera: Camera = PerspectiveCamera()

        #ctx = moderngl.create_context(standalone=True)
        #self.context_wrapper: ContextWrapper = ContextWrapper(ctx)
        #ctx.enable(moderngl.DEPTH_TEST)
        #ctx.enable(moderngl.BLEND)
        #fbo = ctx.simple_framebuffer((960, 540))
        #fbo.use()
        #fbo.clear(0.0, 0.0, 0.0, 1.0)  # background color
        #self.ctx: moderngl.Context = ctx
        #self.fbo: moderngl.Framebuffer = fbo

    def render_scene(self, dt: float):
        for mobject in self.get_descendents():
            mobject.update_dt(dt)
            mobject._camera_ = self.camera
            mobject.render()
        return self

    def wait(self, t: Real):
        window = self.window
        FPS = 30.0
        dt = 1.0 / FPS
        elapsed_time = 0.0
        timestamp = time.time()
        if window is not None:
            while not window.is_closing and elapsed_time < t:
                elapsed_time += dt
                delta_t = time.time() - timestamp
                if dt > delta_t:
                    time.sleep(dt - delta_t)
                timestamp = time.time()
                window.clear()
                self.render_scene(dt)
                window.swap_buffers()
        else:  # TODO
            self.render_scene(dt)
        return self
