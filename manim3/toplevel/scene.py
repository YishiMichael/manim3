from __future__ import annotations


from typing import Self

from ..animatables.arrays.animatable_color import AnimatableColor
from ..animatables.lights.ambient_light import AmbientLight
from ..animatables.camera import Camera
from ..animatables.lighting import Lighting
from ..constants.custom_typing import ColorType
from ..mobjects.mobject import Mobject
from ..timelines.timeline import Timeline
from .toplevel import Toplevel


class Scene(Timeline):
    __slots__ = (
        "_camera",
        "_lighting",
        "_background_color",
        "_background_opacity",
        "_root_mobject",
        "_scene_time"
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
        self._scene_time: float = 0.0
        self.set(
            background_color=Toplevel._get_config().background_color,
            background_opacity=Toplevel._get_config().background_opacity
        )

    def _run(
        self: Self
    ) -> None:
        self.schedule(parent_absolute_rate=lambda: Toplevel._get_scene()._scene_time)
        for scene_time in Toplevel._get_timer().frame_clock():
            self._scene_time = scene_time
            Toplevel._get_window()._pyglet_window.dispatch_events()
            self._progress()
            Toplevel._get_window().clear_event_info_queue()
            Toplevel._get_renderer().process_frame()
            if self.terminated():
                break

    def run(
        self: Self
    ) -> None:
        Toplevel._scene = self
        try:
            self._run()
        except KeyboardInterrupt:
            pass
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
            red, green, blue = AnimatableColor._color_to_array(background_color)
            self._background_color = (float(red), float(green), float(blue))
        if background_opacity is not None:
            self._background_opacity = background_opacity
        return self
