__all__ = ["SceneMobject"]


#import moderngl
#from moderngl_window.context.pyglet.window import Window as PygletWindow
import skia

#from ..cameras.camera import Camera  # TODO: move to a proper location
from ..mobjects.skia_mobject import SkiaMobject
from ..constants import PIXEL_PER_UNIT
from ..custom_typing import *
from ..scene import Scene
#from ..shader_utils import ContextWrapper  # TODO: move to a proper location


class SceneMobject(SkiaMobject):
    def __init__(
        self,
        #camera: Camera,
        scene: Scene,
        frame: skia.Rect,
        #resolution: tuple[int, int],
        #window: PygletWindow | None = None
    ):
        #if window is None:
        #    ctx = moderngl.create_context(standalone=True)
        #else:
        #    ctx = window.ctx
        #self.camera: Camera = camera
        #self.context_wrapper: ContextWrapper = ContextWrapper(ctx)
        super().__init__(
            frame=frame,
            resolution=(
                int(frame.width() * PIXEL_PER_UNIT),
                int(-frame.height() * PIXEL_PER_UNIT)  # flip y
            )
        )
        self.scene: Scene = scene

    def draw(self, canvas: skia.Canvas) -> None:
        scene = self.scene
        fbo = scene.fbo
        #print(bool(fbo.read().lstrip(b"\x00")))
        print("Called SceneMobject")
        scene.render()
        #print(bool(fbo.read().lstrip(b"\x00")))
        #for mobject in self.get_descendents():
        #    shader_data = mobject.setup_shader_data(self.camera)
        #    if shader_data is None:
        #        continue
        #    self.context_wrapper.render(shader_data)
        #print(len(fbo.read(
        #        #viewport=fbo.viewport,
        #        components=4
        #    )))
        #print(fbo.read(
        #        viewport=fbo.viewport,
        #        components=4
        #    ))
        print("After Called SceneMobject", len(fbo.read().strip(b"\x00")))
        canvas.writePixels(
            info=skia.ImageInfo.Make(
                width=self.resolution[0],
                height=self.resolution[1],
                ct=skia.kRGBA_8888_ColorType,
                at=skia.kUnpremul_AlphaType
            ),
            pixels=fbo.read(
                #viewport=fbo.viewport,
                #components=4
            )
        )
