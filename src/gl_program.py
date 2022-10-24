from __future__ import annotations

import itertools as it
import struct

from colour import Color
import moderngl
import numpy as np
from PIL import Image

from cameras.camera import Camera
from cameras.perspective_camera import PerspectiveCamera
from fog import Fog
from geometries.geometry import SphereGeometry
from lights.light import Light
from lights.lights import Lights
#from materials.material import *
from materials.material import MeshBasicMaterial
from mobject import Mesh, Mobject, ShaderData
#from shader_lib import *
from utils.arrays import Mat3, Mat4, Vec2, Vec3, Vec4
from utils.texture import Texture


class Scene:
    def __init__(self):
        self.mobject_node: Mobject = Mobject()
        self.camera: Camera = PerspectiveCamera()
        self.lights: Lights = Lights()
        self.fog: Fog = None
        #self.environment: Any | None = None  # TODO

        ctx = moderngl.create_context(standalone=True)
        ctx.enable(moderngl.DEPTH_TEST)
        #ctx.enable(moderngl.BLEND)
        fbo = ctx.simple_framebuffer((960, 540))
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)  # background color
        self.ctx = ctx
        self.fbo = fbo

    def add(self, *mobjects: Mobject):
        self.mobject_node.add(*mobjects)
        return self

    def add_light(self, *lights: Light):
        for light in lights:
            self.lights.add_light(light)
        return self

    def set_fog(self, fog: Fog):
        self.fog = fog
        return self

    def render(self):
        lights_state = self.lights.setup_lights(self.camera.matrix.inverse())
        # TODO: categorize lights, fog, etc.
        for mobject in self.mobject_node.iter_descendents():
            try:
                shader_data = mobject.setup_shader_data(self.camera, lights_state, self.fog)
            except NotImplementedError:
                continue
            self.render_shader(shader_data)

        fbo = self.fbo
        Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).show()

    def render_shader(self, shader_data: ShaderData):
        ctx = self.ctx
        program = ctx.program(
            vertex_shader=shader_data.vertex_shader,
            fragment_shader=shader_data.fragment_shader
        )

        #attribute_keys = [
        #    k for k, v in program._members.items()
        #    if isinstance(v, moderngl.Attribute)
        #]
        #uniform_keys = [
        #    k for k, v in program._members.items()
        #    if isinstance(v, moderngl.Uniform)
        #]
        loc = 0
        for uniform_name, uniform in shader_data.uniforms.items():
            #print(k, v)
            #if k not in uniform_keys:
            #    continue
            if uniform is None:
                raise ValueError(f"Uniform '{uniform_name}' undefined")
            elif isinstance(uniform, (Mat3, Mat4, Vec2, Vec3, Vec4)):
                program[uniform_name] = tuple(uniform)
            elif isinstance(uniform, Color):
                program[uniform_name] = uniform.rgb
            elif isinstance(uniform, Texture):
                program[uniform_name].__setattr__("value", loc)
                im = Image.open(uniform.url).convert("RGBA")
                texture = ctx.texture(
                    size=im.size,
                    components=len(im.getbands()),
                    data=im.tobytes(),
                )
                texture.use(location=loc)
                loc += 1
            else:
                #print(k, v)
                program[uniform_name] = uniform

        content = []
        for attribute_name, attribute in shader_data.attributes.items():
            attr = attribute[0]
            if isinstance(attr, (Mat3, Mat4, Vec2, Vec3, Vec4)):
                vert_format = f"{len(attr)}f"
                attrs = list(it.chain(*attribute))
            elif isinstance(attr, (float, int)):
                vert_format = "f"
                attrs = list(attribute)
            else:
                raise ValueError(f"Unsupported attribute type '{type(attr)}'")

            #print(vert_format, attrs)
            content.append((
                ctx.buffer(struct.pack(f"{len(attrs)}f", *attrs)),
                vert_format,
                attribute_name
            ))

        #vbo = ctx.buffer(vertex_attribute_items.astype('f4').tobytes())
        ibo = ctx.buffer(np.array(geometry.index).astype("i4").tobytes())
        vao = ctx.vertex_array(
            program=program,
            content=content,
            index_buffer=ibo,
        )
        vao.render(shader_data.render_primitive)
        return self


if __name__ == "__main__":
    scene = Scene()
    scene.camera.apply_matrix(Mat4().rotate(np.pi, Vec3(0, 1, 0))).shift(Vec3(0, 0, 3)).scale(Vec3(1, -1, 1))
    geometry = SphereGeometry()
    #geometry = PlaneGeometry(0.6, 0.8)
    material = MeshBasicMaterial(
        map=Texture("earth_texture.jfif")
    )
    mobject = Mesh(geometry, material)
    scene.add(mobject)
    scene.render()



    #scene = Scene()
    #scene.camera.shift(Vec3(0, 0, 6))
    #path = pathops.Path()
    #path.fillType = pathops.FillType.EVEN_ODD  # TODO: this leads to buggy output image
    #path.moveTo(0, 2)
    #path.lineTo(1, -1)
    #path.lineTo(-1, 1)
    #path.lineTo(1, 1)
    #path.lineTo(-1, -1)
    #path.close()
    #geometry = PathGeometry(path)
    #mobject = Mesh(geometry, MeshBasicMaterial(color=Color("yellow")))
    #mobject.scale(0.5)
    ##path.stroke(4, pathops.LineCap.ROUND_CAP, pathops.LineJoin.ROUND_JOIN, 2)  # TODO: pathops._pathops.UnsupportedVerbError: CONIC
    #path.stroke(0.05, pathops.LineCap.SQUARE_CAP, pathops.LineJoin.BEVEL_JOIN, 0)
    #geometry2 = PathGeometry(path)
    #mobject2 = Mesh(geometry2, MeshBasicMaterial(color=Color("blue")))
    #mobject2.scale(0.5)
    #scene.add(mobject2, mobject)  # reversed?
    ##scene.add(mobject)
    #scene.render()


    """
    sc = Scene()
    sc.add(Group(
        Rectangle(),
        Circle().shift(RIGHT * 3),
    ), Text("Hello!"))
    csc = Scene()
    csc.add(Cube())
    sc.add_scene(csc)  # ?
    sc.render()
    # sc.render_fig(stop_anim=None, stop_time=None)
    # sc.render_video(start_anim=None, stop_anim=None, start_time=None, stop_time=None)
    """
