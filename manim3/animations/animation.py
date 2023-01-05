#__all__ = ["Animation"]


from typing import Generic, TypeVar

from ..custom_typing import Real
from ..mobjects.mobject import Mobject
from ..mobjects.path_mobject import PathMobject


MobjectType = TypeVar("MobjectType", bound="Mobject")


class Animation(Generic[MobjectType]):
    #def __init__(
    #    self,
    #    #run_time: Real = 1.0
    #):
    #    #self.run_time: Real = 1.0
    #    self.elapsed_time: Real = 0.0

    def update_dt(self, mobject: MobjectType, dt: Real) -> None:
        self.elapsed_time += dt
        self.update(mobject, self.elapsed_time)

    def start(self, initial_mobject: MobjectType) -> None:
        self.elapsed_time: Real = 0.0  # TODO: outside __init__
        #self.initial_mobject = mobject.copy()  # TODO: outside __init__

    def update(self, mobject: MobjectType, t: Real) -> None:
        pass

    def expired(self) -> bool:
        return False


class DrawPath(Animation[PathMobject]):
    def start(self, initial_mobject: PathMobject) -> None:
        super().start(initial_mobject)
        import copy
        self.initial_path = copy.deepcopy(initial_mobject._path_)

    def update(self, mobject: PathMobject, t: Real) -> None:
        if t > 3.0:
            return
        #self.initial_mobject.path.partial_by_l_ratio(t / 3).skia_path.dump()
        #self.initial_path.skia_path.dump()
        #print(t)
        #print(mobject._path_._l_final_)
        mobject.set_path(self.initial_path.partial_by_l_ratio(t / 3.0))

    def expired(self) -> bool:
        return self.elapsed_time > 3.0
