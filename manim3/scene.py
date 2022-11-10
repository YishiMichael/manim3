import time

from moderngl_window.context.pyglet.window import Window as PygletWindow

from .cameras.camera import Camera
from .cameras.perspective_camera import PerspectiveCamera
from .mobjects.mobject import Mobject
from .shader_utils import ContextWrapper
from .constants import PIXEL_HEIGHT, PIXEL_WIDTH
from .typing import *


class Scene(Mobject):
    def __init__(self: Self):
        super().__init__()
        window = PygletWindow(
            size=(PIXEL_WIDTH // 2, PIXEL_HEIGHT // 2),  # TODO
            fullscreen=False,
            resizable=True,
            gl_version=(3, 3),
            vsync=True,
            cursor=True
        )
        self.window: PygletWindow = window
        self.context_wrapper: ContextWrapper = ContextWrapper(window.ctx)
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

    def render(self: Self) -> Self:
        for mobject in self.get_descendents():
            shader_data = mobject.setup_shader_data(self.camera)
            if shader_data is None:
                continue
            self.context_wrapper.render(shader_data)
        return self

    def run(self: Self) -> Self:
        window = self.window
        FPS = 10.0
        while not window.is_closing:
            time.sleep(1.0 / FPS)
            window.clear()
            self.render()
            window.swap_buffers()
        return self

    """
    def render_shader(self: Self, shader_data: ShaderData) -> Self:
        ctx = self.ctx
        program = ctx.program(
            vertex_shader=shader_data.vertex_shader,
            fragment_shader=shader_data.fragment_shader
        )

        for uniform_name, uniform in shader_data.uniforms.items():
            if isinstance(uniform, (int, float)):
                program[uniform_name] = uniform
            elif isinstance(uniform, (Mat3, Mat4, Vec2, Vec3, Vec4)):
                program[uniform_name] = tuple(uniform)
            else:
                raise TypeError(f"Cannot handle uniform type '{type(uniform)}'")

        content = []
        for attribute_name, attribute in shader_data.vertex_attributes.items():
            attr = attribute[0]
            if isinstance(attr, (float, int)):
                vert_format = "f"
                attrs = list(attribute)
            elif isinstance(attr, (Mat3, Mat4, Vec2, Vec3, Vec4)):
                vert_format = f"{len(attr)}f"
                attrs = list(it.chain(*attribute))
            else:
                raise ValueError(f"Unsupported attribute type '{type(attr)}'")

            #print(vert_format, attrs)
            content.append((
                ctx.buffer(struct.pack(f"{len(attrs)}f", *attrs)),
                vert_format,
                attribute_name
            ))

        #vbo = ctx.buffer(vertex_attribute_items.astype('f4').tobytes())
        ibo = ctx.buffer(np.array(shader_data.vertex_indices).astype("i4").tobytes())
        vao = ctx.vertex_array(
            program=program,
            content=content,
            index_buffer=ibo,
        )
        vao.render(shader_data.render_primitive)
        return self
    """
