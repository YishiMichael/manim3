from typing import Callable
import weakref

from ..scene.scene import Scene
from ..scene.timeline import Timeline
from ..utils.rate import RateUtils


class Animation(Timeline):
    __slots__ = ("_scene_ref",)

    def __init__(
        self,
        updater: Callable[[float], None] | None = None,
        run_time: float | None = None,
        relative_rate: Callable[[float], float] = RateUtils.linear
    ) -> None:
        super().__init__(
            updater=updater,
            run_time=run_time,
            relative_rate=relative_rate
        )
        self._scene_ref: weakref.ref[Scene] | None = None

    def _is_prepared(
        self,
        timeline: Timeline
    ) -> None:
        if isinstance(timeline, Scene):
            self._scene_ref = weakref.ref(timeline)

    # Access the scene the animation is operated on.
    @property
    def scene(self) -> Scene:
        assert (scene_ref := self._scene_ref) is not None
        assert (scene := scene_ref()) is not None
        return scene
