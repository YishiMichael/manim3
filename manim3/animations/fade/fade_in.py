from ...mobjects.mobject import Mobject
from ..animation.animation import Animation
from ..animation.rates import Rates

#from ..transform.transform_from_copy import TransformFromCopy


#_MobjectT = TypeVar("_MobjectT", bound=Mobject)


class FadeIn(Animation):
    __slots__ = ("_mobject",)

    def __init__(
        self,
        mobject: Mobject
    ) -> None:
        super().__init__()
        self._mobject: Mobject = mobject
        #super().__init__(
        #    mobject=mobject,
        #    func=lambda mob: func(mob.set(opacity=0.0))
        #)

    async def timeline(self) -> None:
        mobject = self._mobject

        self.scene.add(mobject)
        await self.play(mobject.animate.set(opacity=0.0).build(rate=Rates.rewind()))
