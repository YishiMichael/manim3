from ...mobjects.mobject import Mobject
from ..animation.animation import Animation

#from ..transform.transform_from_copy import TransformFromCopy


#_MobjectT = TypeVar("_MobjectT", bound=Mobject)


class FadeIn(Animation):
    __slots__ = (
        "_animation",
        "_mobject"
    )

    def __init__(
        self,
        mobject: Mobject
    ) -> None:
        super().__init__(run_alpha=1.0)
        self._animation: Animation = mobject.animate.set(opacity=0.0).build(rewind=True)
        self._mobject: Mobject = mobject
        #super().__init__(
        #    mobject=mobject,
        #    func=lambda mob: func(mob.set(opacity=0.0))
        #)

    async def timeline(self) -> None:
        self.scene.add(self._mobject)
        await self.play(self._animation)
