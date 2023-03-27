__all__ = ["PlayScene"]


from ..animations.animation import Animation
from ..mobjects.scene_mobject import SceneMobject


class PlayScene(Animation):
    __slots__ = ()

    def __init__(
        self,
        scene_mobject: SceneMobject,
        play_time: float | None = None,
        play_speed: float = 1.0
    ) -> None:
        super().__init__(
            animate_func=animate_func,
            time_regroup_items=[],
        )
