__all__ = ["Animation"]


from typing import Callable


from ..custom_typing import Real
from ..mobjects.mobject import Mobject
#from ..mobjects.shape_mobject import ShapeMobject


class Animation:
    def __init__(
        self,
        animate_func: Callable[[Real], None],
        # `tuple[Real, Mobject, Mobject | None]` stands for a (time, mobject, parent) triplet. `None` represents the scene.
        mobject_add_items: list[tuple[Real, Mobject, Mobject | None]],
        mobject_remove_items: list[tuple[Real, Mobject, Mobject | None]],
        start_time: Real,
        stop_time: Real | None
    ):
        assert stop_time is None or stop_time >= start_time
        self._animate_func: Callable[[Real], None] = animate_func
        self._mobject_add_items: list[tuple[Real, Mobject, Mobject | None]] = mobject_add_items
        self._mobject_remove_items: list[tuple[Real, Mobject, Mobject | None]] = mobject_remove_items
        self._start_time: Real = start_time
        self._stop_time: Real | None = stop_time



#class Animation(Generic[MobjectType]):
#    #def __init__(
#    #    self,
#    #    #run_time: Real = 1.0
#    #):
#    #    #self.run_time: Real = 1.0
#    #    self.elapsed_time: Real = 0.0

#    def update_dt(self, mobject: MobjectType, dt: Real) -> None:
#        self.elapsed_time += dt
#        self.update(mobject, self.elapsed_time)

#    def start(self, initial_mobject: MobjectType) -> None:
#        self.elapsed_time: Real = 0.0  # TODO: outside __init__
#        #self.initial_mobject = mobject.copy()  # TODO: outside __init__

#    def update(self, mobject: MobjectType, t: Real) -> None:
#        pass

#    def expired(self) -> bool:
#        return False


#class DrawShape(Animation[ShapeMobject]):
#    def start(self, initial_mobject: ShapeMobject) -> None:
#        super().start(initial_mobject)
#        self.initial_shape = initial_mobject._shape_

#    def update(self, mobject: ShapeMobject, t: Real) -> None:
#        if t > 3.0:
#            return
#        #self.initial_mobject.path.partial_by_l_ratio(t / 3).skia_path.dump()
#        #self.initial_path.skia_path.dump()
#        #print(t)
#        #print(mobject._path_._l_final_)
#        mobject.set_shape(self.initial_shape.partial(0.0, t / 3.0))

#    def expired(self) -> bool:
#        return self.elapsed_time > 3.0
