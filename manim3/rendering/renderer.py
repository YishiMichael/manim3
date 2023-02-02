__all__ = ["Renderer"]


import os
import subprocess as sp

from PIL import Image

from ..scenes.active_scene_data import ActiveSceneDataSingleton
from ..scenes.scene import Scene
from ..rendering.config import (
    Config,
    ConfigSingleton
)
from ..rendering.render_procedure import RenderProcedure


class Renderer:
    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            config = Config()
        self._config: Config = config

    def run(self, scene_cls: type[Scene]) -> None:
        ConfigSingleton.set(self._config)
        if ConfigSingleton().write_video:
            writing_process = sp.Popen([
                "ffmpeg",
                "-y",  # overwrite output file if it exists
                "-f", "rawvideo",
                "-s", "{}x{}".format(*ConfigSingleton().pixel_size),  # size of one frame
                "-pix_fmt", "rgba",
                "-r", str(ConfigSingleton().fps),  # frames per second
                "-i", "-",  # The input comes from a pipe
                "-vf", "vflip",
                "-an",
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                os.path.join(ConfigSingleton().output_dir, "result.mp4")
            ], stdin=sp.PIPE)
        else:
            writing_process = None

        with RenderProcedure.texture() as color_texture, \
                RenderProcedure.framebuffer(
                    color_attachments=[color_texture],
                    depth_attachment=None
                ) as framebuffer:
            ActiveSceneDataSingleton.set(
                color_texture=color_texture,
                framebuffer=framebuffer,
                writing_process=writing_process
            )

            scene = scene_cls()
            scene.construct()

            # Ensure at least one frame is rendered
            if scene._previous_rendering_timestamp is None:
                scene._render_frame()

            if ConfigSingleton().write_video:
                assert writing_process is not None and writing_process.stdin is not None
                writing_process.stdin.close()
                writing_process.wait()
                writing_process.terminate()
            if ConfigSingleton().write_last_frame:
                # TODO: the image is flipped in y direction
                image = Image.frombytes(
                    "RGBA",
                    ConfigSingleton().pixel_size,
                    framebuffer.read(components=4),
                    "raw"
                )
                image.save(os.path.join(ConfigSingleton().output_dir, "result.png"))
            #if ConfigSingleton().halt_on_last_frame and ConfigSingleton().preview:
            #    assert (window := ContextSingleton._WINDOW) is not None
            #    spf = 1.0 / ConfigSingleton().fps
            #    while not window.is_closing:
            #        time.sleep(spf)
