from typing import Generic, TypeVar

from ..mobjects.mobject import Mobject
from ..mobjects.path_mobject import PathMobject
from ..custom_typing import *


#__all__ = ["Animation"]


MobjectType = TypeVar("MobjectType", bound="Mobject")


class Animation(Generic[MobjectType]):
    #def __init__(
    #    self: Self,
    #    #run_time: Real = 1.0
    #):
    #    #self.run_time: Real = 1.0
    #    self.elapsed_time: Real = 0.0

    def update_dt(self: Self, mobject: MobjectType, dt: Real) -> None:
        self.elapsed_time += dt
        self.update(mobject, self.elapsed_time)

    def start(self: Self, mobject: MobjectType) -> None:
        self.elapsed_time: Real = 0.0  # TODO: outside __init__
        self.initial_mobject = mobject.copy()  # TODO: outside __init__

    def update(self: Self, mobject: MobjectType, t: Real) -> None:
        pass

    def expired(self: Self) -> bool:
        return False


class DrawPath(Animation[PathMobject]):
    def update(self: Self, mobject: PathMobject, t: Real) -> None:
        if t > 3.0:
            t = 3.0
        #self.initial_mobject.path.partial_by_l_ratio(t / 3).skia_path.dump()
        #print(t)
        mobject.set_path(self.initial_mobject.path.partial_by_l_ratio(t / 3.0))

    def expired(self: Self) -> bool:
        return self.elapsed_time > 3.0
