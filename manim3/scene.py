import time

import moderngl
from moderngl_window.context.pyglet.window import Window as PygletWindow

from .cameras.camera import Camera
from .cameras.perspective_camera import PerspectiveCamera
from .mobjects.mobject import Mobject
#from ..mobjects.scene_mobject import SceneMobject
from .utils.renderable import ContextSingleton
from .constants import PIXEL_HEIGHT, PIXEL_WIDTH
from .custom_typing import *


class Scene(Mobject):
    def __init__(self: Self, has_window: bool = True):
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

        ContextSingleton._instance = ctx
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

    def render_scene(self: Self, dt: float) -> Self:
        #shader_data = self.scene_mobject.setup_shader_data(self.scene_mobject.camera)
        #self.scene_mobject.context_wrapper.render(shader_data)
        #return self
        #print()
        #print("Before")
        #print(len(self.fbo.read().lstrip(b"\x00")))
        #print()
        #print(len(self.get_descendents()))
        #print()
        #t = time.time()
        for mobject in self.get_descendents():
            mobject.update(dt)
            mobject._camera = self.camera
            #shader_data = mobject.setup_shader_data()
            mobject.render()
            #if mobject.shader_data is None:
            #    continue
            #self.context_wrapper.render(shader_data)
        #print(time.time()-t)
        #print("After")
        #print(len(self.fbo.read().lstrip(b"\x00")))
        return self

    def wait(self: Self, t: Real) -> Self:
        window = self.window
        FPS = 10.0
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
                #print()
                #print(len(self.fbo.read().lstrip(b"\x00")))
                self.render_scene(dt)
                #print(len(self.fbo.read().lstrip(b"\x00")))
                window.swap_buffers()
        else:  # TODO
            #print()
            #self.fbo.clear()
            #print(123)
            #print(len(self.fbo.read().lstrip(b"\x00")))
            self.render_scene(dt)
            #print(len(self.fbo.read().lstrip(b"\x00")))
        return self
