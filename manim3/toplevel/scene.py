from __future__ import annotations


import asyncio
import itertools
from typing import Self

from ..animatables.lights.ambient_light import AmbientLight
from ..animatables.camera import Camera
from ..animatables.lighting import Lighting
from ..constants.custom_typing import ColorType
from ..mobjects.mobject import Mobject
from ..timelines.timeline.timeline import Timeline
from ..utils.color_utils import ColorUtils
from .toplevel import Toplevel


class Scene(Timeline):
    __slots__ = (
        "_camera",
        "_lighting",
        "_background_color",
        "_background_opacity",
        "_root_mobject",
        "_scene_timer"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._camera: Camera = Camera()
        self._lighting: Lighting = Lighting(AmbientLight())
        self._background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._background_opacity: float = 0.0
        self._root_mobject: Mobject = Mobject()
        self._scene_timer: float = 0.0
        self.set(
            background_color=Toplevel._get_config().background_color,
            background_opacity=Toplevel._get_config().background_opacity
        )

    async def _run_frame(
        self: Self
    ) -> None:
        await asyncio.sleep(0.0)
        Toplevel._get_window()._pyglet_window.dispatch_events()
        self._progress()
        Toplevel._get_window().clear_event_queue()
        Toplevel._get_renderer().process_frame(self)

    async def _run_scene(
        self: Self
    ) -> None:
        self._root_schedule()
        spf = 1.0 / Toplevel._get_config().fps
        for frame_index in itertools.count():
            self._scene_timer = frame_index * spf
            Toplevel._get_logger()._fps_counter.increment_frame()
            Toplevel._get_logger()._scene_timer = self._scene_timer

            async with asyncio.TaskGroup() as task_group:
                task_group.create_task(self._run_frame())
                if Toplevel._get_renderer()._livestream:
                    task_group.create_task(asyncio.sleep(spf))

            if self.get_after_terminated_state() is not None:
                break

    def run(
        self: Self
    ) -> None:
        Toplevel._scene = self
        Toplevel._get_logger()._scene_name = type(self).__name__
        Toplevel._get_logger()._scene_timer = 0.0
        asyncio.run(self._run_scene())
        Toplevel._get_logger()._scene_timer = None
        Toplevel._get_logger()._scene_name = None
        Toplevel._scene = None

    # Shortcut access to root mobject.

    def add(
        self: Self,
        *mobjects: Mobject
    ) -> Self:
        self._root_mobject.add(*mobjects)
        return self

    def discard(
        self: Self,
        *mobjects: Mobject
    ) -> Self:
        self._root_mobject.discard(*mobjects)
        return self

    def clear(
        self: Self
    ) -> Self:
        self._root_mobject.clear()
        return self

    @property
    def camera(
        self: Self
    ) -> Camera:
        return self._camera

    @property
    def lighting(
        self: Self
    ) -> Lighting:
        return self._lighting

    def bind_camera(
        self: Self,
        camera: Camera,
        *,
        broadcast: bool = True
    ) -> Self:
        self._camera = camera
        self._root_mobject.bind_camera(camera, broadcast=broadcast)
        return self

    def bind_lighting(
        self: Self,
        lighting: Lighting,
        *,
        broadcast: bool = True
    ) -> Self:
        self._lighting = lighting
        self._root_mobject.bind_lighting(lighting, broadcast=broadcast)
        return self

    def set(
        self: Self,
        camera: Camera | None = None,
        lighting: Lighting | None = None,
        background_color: ColorType | None = None,
        background_opacity: float | None = None,
        *,
        broadcast: bool = True
    ) -> Self:
        if camera is not None:
            self.bind_camera(camera, broadcast=broadcast)
        if lighting is not None:
            self.bind_lighting(lighting, broadcast=broadcast)
        if background_color is not None:
            red, green, blue = tuple(float(component) for component in ColorUtils.color_to_array(background_color))
            self._background_color = (red, green, blue)
        if background_opacity is not None:
            self._background_opacity = background_opacity
        return self
